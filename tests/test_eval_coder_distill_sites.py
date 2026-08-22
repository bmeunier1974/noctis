"""The two non-contract-typed site declarations (#192): the coder and the memory distiller.

Both sites are declarations *over production code*, so every test here asks the one question a
declaration can get wrong: does it still name what production actually does? The renderer tests
therefore never inspect the adapter — they drive the real engine (``StrategyAuthor.author`` with a
fake coder client, ``distill_findings`` with a fake distill client), take the exact prompt the
transport was handed, and assert the declared renderer reproduces it **byte for byte**. A rendered
copy that drifted from production would fail here rather than quietly benchmark a prompt nobody
ships.

The knob tests are the other half: a benchmark override may only name a lever that exists today, so
each declared knob is checked for membership in the settings model it mirrors, and a plausible
future knob (``temperature``) is checked to be refused rather than silently accepted.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from noctis.config.settings import AgentResearchConfig, ResearchConfig
from noctis.eval.coder_distill_sites import (
    CODER_SITE,
    DISTILL_SITE,
    NO_PROMPT,
    AuthoringJob,
    CoderKnobs,
    DistillKnobs,
    FindingsHistory,
)
from noctis.eval.harness import HarnessSpec
from noctis.eval.knobs import UnknownKnob
from noctis.eval.registry import site
from noctis.memory import InMemoryMemory
from noctis.research.author import WORKED_EXAMPLE_NAME, StrategyAuthor, StrategyBrief
from noctis.research.distill import distill_findings
from noctis.strategies.families import FamilyRegistry
from tests.test_distill import FakeDistillClient
from tests.test_research_tools import PROBE
from tests.test_strategy_author import BRIEF, FakeCoder, fenced

_REPO_SEEDS = Path(__file__).resolve().parents[1] / "strategies"


@pytest.fixture
def families() -> FamilyRegistry:
    return FamilyRegistry()


@pytest.fixture
def seeds(tmp_path: Path) -> Path:
    """A library root carrying the committed worked-example seed, as a real install has."""
    shutil.copyfile(
        _REPO_SEEDS / f"{WORKED_EXAMPLE_NAME}.py", tmp_path / f"{WORKED_EXAMPLE_NAME}.py"
    )
    return tmp_path


def _findings(count: int = 12) -> InMemoryMemory:
    memory = InMemoryMemory()
    for index in range(count):
        memory.append_finding(f"REJECTED strategy s{index} — lesson {index}")
    return memory


# ── the declarations resolve from the shipped registry ────────────────────────────────────────
def test_the_coder_site_is_declared_with_no_emit_contract() -> None:
    """The coder answers with a fenced code block the write gate judges — not a typed record."""
    declared = site("coder")

    assert declared is CODER_SITE
    assert declared.contract is None


def test_the_distill_site_resolves_from_the_shipped_registry_under_its_own_id() -> None:
    declared = site("distill")

    assert declared is DISTILL_SITE
    assert declared.id == "distill"


def test_both_declarations_carry_a_version_a_record_can_compare_generations_on() -> None:
    assert CODER_SITE.version and DISTILL_SITE.version


# ── the knob sets name only settings that exist today ─────────────────────────────────────────
def test_every_coder_knob_names_a_research_setting_that_exists_today() -> None:
    """A declared knob is a *reference* to a production lever: no knob is invented here."""
    assert CoderKnobs.knob_names() <= frozenset(AgentResearchConfig.model_fields)


def test_the_coder_knob_set_names_the_whole_coder_model_settings_family() -> None:
    """Including the sampling pair and the retry budget production grew in #222."""
    assert CoderKnobs.knob_names() == frozenset(
        {
            "coder_model",
            "coder_fallback_model",
            "max_escalations",
            "coder_thinking",
            "coder_fallback_thinking",
            "coder_max_tokens",
            "coder_temperature",
            "coder_seed",
            "coder_retries",
            "max_author_calls",
        }
    )


def test_the_coder_knob_defaults_are_the_shipped_research_settings_defaults() -> None:
    shipped = AgentResearchConfig()
    knobs = CoderKnobs()

    assert all(getattr(knobs, name) == getattr(shipped, name) for name in CoderKnobs.knob_names())


def test_a_bench_override_naming_a_knob_the_coder_does_not_have_is_refused() -> None:
    """The coder's sampling levers are spelled ``coder_temperature``/``coder_seed``, as the
    settings model spells them — a bare ``temperature`` names no knob this site has."""
    spec = HarnessSpec(knob_overrides={"temperature": 0.7, "seed": 11})

    with pytest.raises(UnknownKnob) as error:
        spec.validate_knobs(CODER_SITE.knobs)

    assert "temperature" in str(error.value) and "seed" in str(error.value)


def test_the_distill_knob_set_is_the_cadence_setting_and_nothing_else() -> None:
    """Distillation's other levers are module constants; a declaration promotes none of them."""
    assert DistillKnobs.knob_names() == frozenset({"memory_distill_every"})
    assert "memory_distill_every" in ResearchConfig.model_fields
    assert DistillKnobs().memory_distill_every == ResearchConfig().memory_distill_every


# ── the renderers reproduce the prompts production sends, byte for byte ───────────────────────
def test_the_coder_renderer_reproduces_the_prompt_production_sends_the_coder(
    seeds, families, fast_gate
) -> None:
    client = FakeCoder([fenced(PROBE)])
    engine = StrategyAuthor(client=client, strategies_dir=seeds, families=families)

    # Render the job the driver is about to pose, then let production pose it for real.
    rendered = CODER_SITE.render(AuthoringJob(engine, "probe", BRIEF), HarnessSpec())
    engine.author("probe", BRIEF)
    sent = client.calls[0]

    assert rendered == f"{sent['system']}\n\n{sent['messages'][0]['content']}"
    # Not a vacuous match: the install-derived prefix and the brief are both really in there.
    assert "COMPLETE WORKED EXAMPLE" in rendered and BRIEF.thesis in rendered


def test_the_coder_renderer_reproduces_a_revision_prompt_for_a_name_that_already_exists(
    seeds, families, fast_gate
) -> None:
    """The adapter resolves the revision/reference material the way production does — by asking
    the engine, never by re-deriving it."""
    first = FakeCoder([fenced(PROBE)])
    StrategyAuthor(client=first, strategies_dir=seeds, families=families).author("probe", BRIEF)

    revision = StrategyBrief(
        thesis=BRIEF.thesis,
        entry_exit="Add a short leg below the mean.",
        param_space=BRIEF.param_space,
        scenarios=BRIEF.scenarios,
    )
    client = FakeCoder([fenced(PROBE)])
    engine = StrategyAuthor(client=client, strategies_dir=seeds, families=families)

    rendered = CODER_SITE.render(AuthoringJob(engine, "probe", revision), HarnessSpec())
    engine.author("probe", revision)
    sent = client.calls[0]

    assert rendered == f"{sent['system']}\n\n{sent['messages'][0]['content']}"
    assert "CURRENT VERSION of probe" in rendered


def test_the_coder_renderer_says_no_prompt_was_composed_for_a_brief_the_engine_refuses(
    seeds, families
) -> None:
    """An unresolvable reference spends no completion in production, so there are no bytes to
    reproduce — the renderer stays total and names the refusal instead of inventing a prompt."""
    engine = StrategyAuthor(client=FakeCoder([]), strategies_dir=seeds, families=families)
    brief = StrategyBrief(
        thesis=BRIEF.thesis,
        entry_exit=BRIEF.entry_exit,
        param_space=BRIEF.param_space,
        scenarios=BRIEF.scenarios,
        reference="kama_adaptive_trend",
    )

    rendered = CODER_SITE.render(AuthoringJob(engine, "probe", brief), HarnessSpec())

    assert rendered.startswith(NO_PROMPT)
    assert "kama_adaptive_trend" in rendered


def test_the_distill_renderer_reproduces_the_prompt_production_sends_the_distiller() -> None:
    memory = _findings()
    client = FakeDistillClient("- a lesson")
    assert distill_findings(memory, client) is True
    sent = client.calls[0]["messages"][0]["content"]

    rendered = DISTILL_SITE.render(FindingsHistory(tuple(memory.findings())), HarnessSpec())

    assert rendered == sent
    assert "lesson 0" in rendered and "lesson 11" in rendered  # the whole history, oldest first


def test_the_distill_renderer_honours_the_line_budget_the_caller_folds_with() -> None:
    """``max_lines`` is a production argument of the distillation call, so the input carries it."""
    memory = _findings()
    client = FakeDistillClient("- a lesson")
    assert distill_findings(memory, client, max_lines=4) is True
    sent = client.calls[0]["messages"][0]["content"]

    rendered = DISTILL_SITE.render(
        FindingsHistory(tuple(memory.findings()), max_lines=4), HarnessSpec()
    )

    assert rendered == sent
