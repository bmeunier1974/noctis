"""The pages that say where the FORMULATE-authored oracle crosses a boundary are held to the
modules that own those crossings (story #324, epic #319).

The epic gave the scenario spec one home: `noctis.strategies.scenario_spec` owns the vocabulary
*and* all three crossings into it — the model dialect, the JSON carrier, and compile — while the
suite-shape rules stay the known-outcome contract's own, written once in
`noctis.strategies.scenarios.check_suite_shape`. That is only half-shipped while the prose still
describes the old shape: a reader told the model's dialect is parsed in `research/driver.py` goes
looking through a 2,000-line episode runner for the parse that moved, and a reviewer asked to hold
"the model never writes a bar index" has never been told the term anywhere.

Four checks, each red on a **code** change rather than on a human remembering:

* every public name the prose spells out resolves on the module it is claimed to live on, so
  renaming `spec_from_payload` or `check_suite_shape` reddens the pages that named it;
* the glossary defines **Scenario spec** — the one module, the three crossings, where the shape
  rules live, the schema as a prompt asset — so the term a review uses resolves to a file;
* the pages a reader lands on (`docs/architecture.md`, `docs/research.md`) name those crossings
  with their full module paths, and say the schema derives from the vocabulary;
* each module docstring states the crossing(s) it owns, and the spec module's states the import
  rule that keeps the strategy layer under the research layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from noctis.observability.prompt_id import SITE_ASSETS
from noctis.strategies import scenario_spec, scenarios

ROOT = Path(__file__).resolve().parents[1]

# The two modules every page must send a reader to, by the names they actually have.
SPEC_MODULE = scenario_spec.__name__  # noctis.strategies.scenario_spec
SCENARIOS_MODULE = scenarios.__name__  # noctis.strategies.scenarios

# Spelled in full: a bare `strategies/scenario_spec.py` would point at the seed folder.
SPEC_PATH = "src/noctis/strategies/scenario_spec.py"
SCENARIOS_PATH = "src/noctis/strategies/scenarios.py"

# The prompt call site the schema is an asset of — model-facing text, ratcheted like a prompt.
PROMPT_SITE = "episodic"

# The three crossings, in the order the prose introduces them: model dialect, carrier, compile.
MODEL_DIALECT = ("spec_from_payload", "SPEC_JSON_SCHEMA", "LEG_KINDS", "PARSE_WARM")
CARRIER = ("spec_to_json", "spec_from_json")
COMPILE = ("compile_spec",)

# Every public name the pages and the docstrings spell out, beside the module it must resolve on.
SPEC_NAMES = (*MODEL_DIALECT, *CARRIER, *COMPILE, "SpecSuite", "SpecError")
SCENARIOS_NAMES = ("check_suite_shape", "check_scenario_contract", "ScenarioError")


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _glossary_entry(title: str) -> str:
    """One ``## <title>`` section of the domain glossary."""
    text = _read("CONTEXT.md")
    start = text.find(f"## {title}\n")
    assert start >= 0, f"CONTEXT.md: the '{title}' entry is gone — retarget this test"
    end = text.find("\n## ", start + 1)
    return text[start : end if end >= 0 else len(text)]


def _block(relative: str, start: str, end: str) -> str:
    """The stretch of a page from its bold lead-in up to the next one."""
    text = _read(relative)
    first = text.find(start)
    assert first >= 0, f"{relative}: no block starting {start!r} — retarget this test"
    stop = text.find(end, first + 1)
    return text[first : stop if stop >= 0 else len(text)]


def _missing(text: str, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if name not in text]


# ─────────────────────────────────────────────────────────────────────────────
# The pin: every name the prose spells out is an attribute of the module it claims
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", SPEC_NAMES)
def test_every_spec_name_the_prose_spells_out_resolves_on_the_spec_module(name: str) -> None:
    """Rename one of these and the pages that named it are describing a module that has no such
    thing — which is what the page assertions below then say out loud."""
    assert hasattr(scenario_spec, name), f"{SPEC_MODULE} has no {name}"


@pytest.mark.parametrize("name", SCENARIOS_NAMES)
def test_every_scenarios_name_the_prose_spells_out_resolves_on_the_contract_module(
    name: str,
) -> None:
    """The shape rules are the contract's, so their names must resolve there and not on the spec."""
    assert hasattr(scenarios, name), f"{SCENARIOS_MODULE} has no {name}"


def test_the_schema_module_really_is_an_asset_of_the_prompt_site_the_glossary_names() -> None:
    """The glossary calls `SPEC_JSON_SCHEMA` a prompt asset; the asset map has to agree."""
    assert SPEC_PATH in SITE_ASSETS[PROMPT_SITE]


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT.md — the glossary entry
# ─────────────────────────────────────────────────────────────────────────────
def test_the_glossary_entry_sits_directly_after_the_strategy_header_entry() -> None:
    """Two halves of one strategy file's record read in order: the header, then its oracle."""
    text = _read("CONTEXT.md")
    header = text.find("## Strategy header\n")
    spec = text.find("## Scenario spec\n")
    assert header >= 0 and spec > header, (
        "CONTEXT.md has no 'Scenario spec' entry after 'Strategy header'"
    )
    assert text.find("\n## ", header + 1) == spec - 1, (
        "CONTEXT.md: 'Scenario spec' must come immediately after 'Strategy header'"
    )


def test_the_glossary_defines_the_scenario_spec_on_the_module_that_owns_it() -> None:
    """A reader of the term reaches the one strategy-layer module, and its three crossings."""
    entry = _glossary_entry("Scenario spec")

    assert SPEC_MODULE in entry
    assert not _missing(entry, (*MODEL_DIALECT, *CARRIER, *COMPILE))


def test_the_glossary_entry_states_the_rule_the_term_exists_for() -> None:
    """The oracle is honest because authorship of the bar arithmetic is inverted."""
    entry = _glossary_entry("Scenario spec")

    assert "never writes a bar index" in entry
    assert "noctis.research" in entry


def test_the_glossary_entry_puts_the_suite_shape_rules_in_the_contracts_own_module() -> None:
    """The one question the split answers: which module refuses a five-tape suite with no exit?"""
    entry = _glossary_entry("Scenario spec")

    assert SCENARIOS_MODULE in entry
    assert not _missing(entry, SCENARIOS_NAMES)


def test_the_glossary_entry_names_the_schema_as_an_asset_of_its_prompt_site() -> None:
    """Model-facing text in the strategy layer still ratchets like a prompt — say so."""
    entry = _glossary_entry("Scenario spec")

    assert "SPEC_JSON_SCHEMA" in entry
    assert f"`{PROMPT_SITE}`" in entry


# ─────────────────────────────────────────────────────────────────────────────
# docs/architecture.md — "Where the spec lives"
# ─────────────────────────────────────────────────────────────────────────────
def test_the_architecture_page_names_every_crossing_and_both_full_module_paths() -> None:
    """The page a maintainer lands on says which module owns each crossing, by path."""
    block = _block("docs/architecture.md", "**Where the spec lives.**", "\n## ")

    assert not _missing(block, ("spec_from_payload", "SPEC_JSON_SCHEMA", "LEG_KINDS"))
    assert "check_suite_shape" in block
    assert SPEC_PATH in block and SCENARIOS_PATH in block


def test_the_architecture_page_keeps_the_purity_claim_the_layering_rests_on() -> None:
    """Strategy layer under research layer: the compiler may not import upward."""
    block = _block("docs/architecture.md", "**Where the spec lives.**", "\n## ")

    assert "noctis.research" in block


# ─────────────────────────────────────────────────────────────────────────────
# docs/research.md — the FORMULATE paragraph
# ─────────────────────────────────────────────────────────────────────────────
def test_the_research_page_says_the_schema_derives_from_the_vocabulary() -> None:
    """The offer and the parse are one list, not two lists somebody keeps in step."""
    block = _block(
        "docs/research.md",
        "**FORMULATE emits a structured",
        "**Compile-failure re-prompt.**",
    )

    assert not _missing(block, ("SPEC_JSON_SCHEMA", "LEG_KINDS", "spec_from_payload"))


def test_the_research_page_says_the_suite_shape_rules_are_the_contracts_own() -> None:
    """A FORMULATE refusal about suite shape comes from the contract, not from the spec."""
    block = _block(
        "docs/research.md",
        "**FORMULATE emits a structured",
        "**Compile-failure re-prompt.**",
    )

    assert "check_suite_shape" in block
    assert SCENARIOS_PATH in block


# ─────────────────────────────────────────────────────────────────────────────
# The module docstrings — each says which crossing it owns
# ─────────────────────────────────────────────────────────────────────────────
def test_the_spec_module_docstring_names_the_three_crossings_it_owns() -> None:
    """Opening the file tells you what it is for before you read a line of code."""
    doc = scenario_spec.__doc__ or ""

    assert not _missing(doc, (*MODEL_DIALECT, *CARRIER, *COMPILE))


def test_the_spec_module_docstring_states_the_import_rule() -> None:
    """The rule that keeps the compiler usable from the write gate's subprocess."""
    doc = scenario_spec.__doc__ or ""

    assert "noctis.research" in doc


def test_the_spec_module_docstring_disclaims_the_rules_it_does_not_own() -> None:
    """It compiles, then hands the compiled tuple to the contract's own arbiter."""
    doc = scenario_spec.__doc__ or ""

    assert "check_suite_shape" in doc


def test_the_scenarios_module_docstring_names_the_one_check_both_dialects_run() -> None:
    """The shape rules have one spelling; the module that holds it must say so."""
    doc = scenarios.__doc__ or ""

    assert "check_suite_shape" in doc
    assert "check_scenario_contract" in doc
    assert f"{SPEC_MODULE}.compile_spec" in doc
