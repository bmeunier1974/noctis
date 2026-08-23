"""The eval layer's boundary types (#190): ``AgentSite``, ``SiteKnobs``, and the id→site registry.

These are declaration-only types — nothing here runs a site, and no site is declared yet (#191 and
#192 do that). So every scenario builds its own miniature site: a dummy input record, a renderer
that composes a prompt from a :class:`~noctis.eval.harness.HarnessSpec`, a knob set, and an
``AgentSite`` tying them together. What is asserted is the external behaviour a benchmark harness
depends on — what a declaration holds, what a knob set refuses, what the registry returns.

The engine's own emit contract type is referenced by identity (``noctis.research.episode``'s
``EmitContract``, the frozen record the episodic driver already emits through), because the point
of a declaration is to name what production uses, never to copy it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import pytest

from noctis.eval.harness import HarnessSpec
from noctis.eval.knobs import SiteKnobs, UnknownKnob
from noctis.eval.registry import (
    DECLARED_SITES,
    SITES,
    UnknownSite,
    index_sites,
    site,
    sites,
)
from noctis.eval.site import AgentSite
from noctis.research.driver import FORMULATE_CONTRACT
from noctis.research.episode import EmitContract


@dataclass(frozen=True)
class _Briefing:
    """A miniature site input: what a builder would be handed to render a prompt from."""

    thesis: str


@dataclass(frozen=True)
class _Verdict:
    """A miniature site output: what a contract's ``parse`` would return."""

    verdict: str


@dataclass(frozen=True)
class _ExampleKnobs(SiteKnobs):
    """A miniature site knob set — the shape a harness validates an override against."""

    model: str = "ollama/qwen3"
    max_tokens: int = 2048


@dataclass(frozen=True)
class _EmptyKnobs(SiteKnobs):
    """A site whose levers are module constants today (the honest near-empty knob set)."""


def _render(briefing: _Briefing, spec: HarnessSpec) -> str:
    """A renderer of the declared shape: input + harness spec in, one prompt string out."""
    parts = [briefing.thesis]
    if spec.contract_sheet:
        parts.append("CONTRACT SHEET")
    return "\n".join(parts)


_CONTRACT: EmitContract[_Verdict] = EmitContract(
    name="emit_verdict",
    description="Emit the verdict.",
    schema={"type": "object", "properties": {"verdict": {"type": "string"}}},
    parse=lambda payload: _Verdict(verdict=str(payload["verdict"])),
)


def _site(site_id: str = "example", version: str = "1") -> AgentSite[_Briefing, _Verdict]:
    return AgentSite(
        id=site_id,
        version=version,
        contract=_CONTRACT,
        render=_render,
        knobs=_ExampleKnobs,
        scorers=(),
    )


# ── AgentSite: one frozen declaration per call site ──────────────────────────────────────
def test_an_agent_site_declares_the_pieces_that_define_one_call_site() -> None:
    declaration = _site()

    assert declaration.id == "example"
    assert declaration.version == "1"
    assert declaration.contract is _CONTRACT
    assert declaration.render is _render
    assert declaration.knobs is _ExampleKnobs


def test_an_agent_site_declaration_cannot_be_reassigned_after_construction() -> None:
    declaration = _site()

    with pytest.raises(dataclasses.FrozenInstanceError):
        declaration.version = "2"  # type: ignore[misc]


def test_a_site_may_declare_no_emit_contract_for_a_gate_judged_call_site() -> None:
    """The coder is free-form codegen judged by the write gate, so its contract is honestly None."""
    declaration: AgentSite[_Briefing, str] = AgentSite(
        id="coder",
        version="1",
        contract=None,
        render=lambda briefing, spec: briefing.thesis,
        knobs=_ExampleKnobs,
        scorers=(),
    )

    assert declaration.contract is None


def test_a_site_declares_the_engines_own_emit_contract_object_rather_than_a_copy() -> None:
    declaration = AgentSite(
        id="formulate",
        version="1",
        contract=FORMULATE_CONTRACT,
        render=lambda briefing, spec: "",
        knobs=_ExampleKnobs,
        scorers=(),
    )

    assert declaration.contract is FORMULATE_CONTRACT


def test_the_scorer_slot_of_a_declared_site_is_empty_until_the_eval_core_lands() -> None:
    assert _site().scorers == ()


def test_a_declared_renderer_composes_its_prompt_from_the_harness_spec_it_is_handed() -> None:
    declaration = _site()

    production = declaration.render(_Briefing(thesis="momentum decays"), HarnessSpec())
    ablated = declaration.render(
        _Briefing(thesis="momentum decays"), HarnessSpec(contract_sheet=False)
    )

    assert production == "momentum decays\nCONTRACT SHEET"
    assert ablated == "momentum decays"


# ── SiteKnobs: the typed knob set an override is validated against ───────────────────────
def test_a_site_knob_set_reports_the_knob_names_it_declares() -> None:
    assert _ExampleKnobs.knob_names() == frozenset({"model", "max_tokens"})


def test_a_site_whose_levers_are_module_constants_declares_an_empty_knob_set() -> None:
    assert _EmptyKnobs.knob_names() == frozenset()


def test_an_override_naming_knobs_the_site_declares_validates() -> None:
    override = {"model": "claude-sonnet-5", "max_tokens": 4096}

    assert _ExampleKnobs.validate_overrides(override) is None


def test_an_empty_override_validates_against_any_knob_set() -> None:
    assert _EmptyKnobs.validate_overrides({}) is None


def test_an_override_naming_a_knob_the_site_does_not_declare_is_refused_by_name() -> None:
    with pytest.raises(UnknownKnob) as error:
        _ExampleKnobs.validate_overrides({"temperature": 0.7})

    message = str(error.value)
    assert "temperature" in message
    assert "_ExampleKnobs" in message


def test_a_refusal_names_every_unknown_knob_in_one_pass() -> None:
    with pytest.raises(UnknownKnob) as error:
        _ExampleKnobs.validate_overrides({"temperature": 0.7, "seed": 11, "model": "x"})

    message = str(error.value)
    assert "seed" in message
    assert "temperature" in message


def test_a_site_knob_set_cannot_be_reassigned_after_construction() -> None:
    knobs = _ExampleKnobs()

    with pytest.raises(dataclasses.FrozenInstanceError):
        knobs.max_tokens = 4096  # type: ignore[misc]


# ── the registry: a plain id→site lookup ─────────────────────────────────────────────────
def test_the_registry_looks_a_site_up_by_the_id_it_declares() -> None:
    declared = _site("formulate")

    registry = index_sites([declared])

    assert site("formulate", registry) is declared


def test_the_registry_lists_every_site_it_indexes_in_declaration_order() -> None:
    registry = index_sites([_site("coder"), _site("formulate"), _site("decide")])

    assert [declared.id for declared in sites(registry)] == ["coder", "formulate", "decide"]


def test_two_sites_declaring_the_same_id_are_refused_by_the_registry() -> None:
    with pytest.raises(ValueError) as error:
        index_sites([_site("decide", version="1"), _site("decide", version="2")])

    assert "decide" in str(error.value)


def test_looking_up_a_site_id_nothing_declares_names_the_ids_that_are_declared() -> None:
    registry = index_sites([_site("coder"), _site("distill")])

    with pytest.raises(UnknownSite) as error:
        site("formulate", registry)

    message = str(error.value)
    assert "formulate" in message
    assert "coder" in message
    assert "distill" in message


def test_every_shipped_declaration_is_indexed_under_its_own_id() -> None:
    """The registry is a plain mapping over the module-level declarations — it invents no ids."""
    assert all(declared.id == site_id for site_id, declared in SITES.items())
    assert len(SITES) == len(DECLARED_SITES)


def test_the_shipped_registry_is_the_index_of_the_shipped_declarations() -> None:
    assert sites() == tuple(DECLARED_SITES)


def test_the_shipped_registry_cannot_be_mutated_by_a_caller() -> None:
    """The registry is declared as code, not registered at runtime: there is no write path."""
    with pytest.raises(TypeError):
        SITES["smuggled"] = _site("smuggled")  # type: ignore[index]


def test_a_scorer_tuple_may_be_typed_against_the_forward_declaration_the_eval_core_will_own() -> (
    None
):
    """``Scorer`` is a placeholder protocol; a site can already be annotated against it."""
    from noctis.eval.site import Scorer

    scorers: tuple[Scorer[_Briefing, _Verdict], ...] = ()
    declaration: AgentSite[_Briefing, _Verdict] = AgentSite(
        id="decide",
        version="1",
        contract=_CONTRACT,
        render=_render,
        knobs=_ExampleKnobs,
        scorers=scorers,
    )

    assert declaration.scorers == ()


def test_a_site_whose_cases_carry_no_difficulty_labels_declares_no_axes() -> None:
    """The honest default: an absent axis vocabulary, not an empty measurement (#306)."""
    assert _site().difficulty_axes == ()


def test_a_declaration_carries_the_axis_vocabulary_a_reading_stratifies_by() -> None:
    declaration = dataclasses.replace(_site(), difficulty_axes=("margin", "evidence_depth"))

    assert declaration.difficulty_axes == ("margin", "evidence_depth")


def test_a_declaration_reads_back_as_the_plain_data_a_harness_looks_up() -> None:
    """Declaration-only: a site is data, so a harness can enumerate it without running anything."""
    fields: dict[str, Any] = {field.name: field for field in dataclasses.fields(_site())}

    assert set(fields) == {
        "id",
        "version",
        "contract",
        "render",
        "knobs",
        "scorers",
        "difficulty_axes",
    }
