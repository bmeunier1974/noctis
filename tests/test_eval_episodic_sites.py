"""The three episodic site declarations (#191): formulate, decide and discover, bound by identity.

A declaration earns its keep only if it names *the production objects* — the very emit contracts
the episodic driver emits through and the very briefing builders it renders its prompts with. So
these tests never inspect how a declaration is put together; they assert the two things a benchmark
harness depends on:

* the contract a site declares **is** the driver's own constant (``is``, never ``==``), and
* the prompt a site's renderer produces is **byte-identical** to calling the production builder on
  the same on-disk state — through a realistic session fixture (the one ``tests/test_briefings.py``
  builds: a populated lake digest, champions, memory, library, journal evidence and a ledger).

The knob sets are checked the same way: every knob a site declares must be a real field of the
research-agent settings model, so no declaration can quietly promote a bench-only lever into a name
that looks like config.
"""

from __future__ import annotations

import pytest

from noctis.config.settings import AgentResearchConfig
from noctis.eval.episodic_sites import (
    DECIDE_CONTRACTS,
    DECIDE_SITE,
    DISCOVER_SITE,
    EPISODIC_SITES,
    FORMULATE_SITE,
    DecideKnobs,
    DecideSiteInput,
    DiscoverKnobs,
    DiscoverSiteInput,
    FormulateKnobs,
    FormulateSiteInput,
)
from noctis.eval.harness import HarnessSpec
from noctis.eval.knobs import SiteKnobs
from noctis.eval.registry import site
from noctis.research.briefings import decide_briefing, discover_briefing, formulate_briefing
from noctis.research.driver import (
    DECIDE_CONTRACT,
    DECIDE_FINAL_CONTRACT,
    DISCOVER_CONTRACT,
    FORMULATE_CONTRACT,
)
from tests.test_briefings import _populate

_WINDOW = 10_000_000

_PROFILE = {"trend": "low", "volatility": "high", "liquidity": "low"}
_FETCH_WINDOW = {"start": "2026-02-08", "end": "2026-03-09", "history_days": 30}
_THESIS = "fade panic in thin names once volatility clears cost"
_CHARACTER = "illiquid volatile small-caps that mean-revert"


@pytest.fixture(autouse=True)
def _in_process_gate(fast_gate):
    """The fixture writes library strategies; these tests assert prompts, not gate isolation."""


# ── the declarations are resolvable, and their contracts are the driver's own objects ────────
def test_the_shipped_registry_resolves_the_three_episodic_sites_by_id() -> None:
    assert site("formulate") is FORMULATE_SITE
    assert site("decide") is DECIDE_SITE
    assert site("discover") is DISCOVER_SITE


def test_the_episodic_sites_are_declared_in_the_shipped_registry() -> None:
    assert [declared.id for declared in EPISODIC_SITES] == ["formulate", "decide", "discover"]


def test_the_formulate_site_declares_the_contract_the_driver_emits_formulations_through() -> None:
    assert FORMULATE_SITE.contract is FORMULATE_CONTRACT


def test_the_decide_site_declares_the_primary_verdict_contract_the_driver_asks_with() -> None:
    assert DECIDE_SITE.contract is DECIDE_CONTRACT


def test_the_discover_site_declares_the_contract_the_driver_asks_for_symbols_with() -> None:
    assert DISCOVER_SITE.contract is DISCOVER_CONTRACT


def test_each_episodic_site_declares_a_hand_bumped_contract_version() -> None:
    assert [declared.version for declared in EPISODIC_SITES] == ["1", "1", "1"]


# ── the decide site's version covers the final-ask variant (#99) ─────────────────────────────
def test_the_decide_declarations_version_covers_the_final_ask_contract_variant() -> None:
    """Both asks ride one version, so the variant is reachable from the declaration by identity."""
    covered = DECIDE_CONTRACTS

    assert any(contract is DECIDE_CONTRACT for contract in covered)
    assert any(contract is DECIDE_FINAL_CONTRACT for contract in covered)
    assert any(contract is DECIDE_SITE.contract for contract in covered)


def test_the_final_ask_variant_drops_revise_from_the_emit_vocabulary_of_the_same_site() -> None:
    """Why one version covers two contracts: they are one ask whose vocabulary narrows (#99)."""
    primary = DECIDE_CONTRACT.schema["properties"]["verdict"]["enum"]
    final = DECIDE_FINAL_CONTRACT.schema["properties"]["verdict"]["enum"]

    assert "revise" in primary
    assert "revise" not in final
    assert set(final) < set(primary)


# ── the renderers forward to the production builders, byte for byte ──────────────────────────
def test_the_formulate_renderer_produces_the_production_builders_briefing_byte_for_byte(
    tmp_path,
) -> None:
    box, ledger, mandate = _populate(tmp_path)
    site_input = FormulateSiteInput(
        toolbox=box, ledger=ledger, context_window=_WINDOW, mandate=mandate
    )

    rendered = FORMULATE_SITE.render(site_input, HarnessSpec())

    assert rendered == formulate_briefing(box, ledger, mandate=mandate, context_window=_WINDOW)


def test_the_decide_renderer_produces_the_production_builders_briefing_byte_for_byte(
    tmp_path,
) -> None:
    box, ledger, mandate = _populate(tmp_path)
    site_input = DecideSiteInput(
        toolbox=box, ledger=ledger, strategy="probe", context_window=_WINDOW, mandate=mandate
    )

    rendered = DECIDE_SITE.render(site_input, HarnessSpec())

    assert rendered == decide_briefing(
        box, ledger, "probe", mandate=mandate, context_window=_WINDOW
    )


def test_the_discover_renderer_produces_the_production_builders_briefing_byte_for_byte(
    tmp_path,
) -> None:
    box, ledger, mandate = _populate(tmp_path)
    site_input = DiscoverSiteInput(
        toolbox=box,
        ledger=ledger,
        thesis=_THESIS,
        symbol_character=_CHARACTER,
        profile=_PROFILE,
        context_window=_WINDOW,
        window=_FETCH_WINDOW,
        mandate=mandate,
    )

    rendered = DISCOVER_SITE.render(site_input, HarnessSpec())

    assert rendered == discover_briefing(
        box,
        ledger,
        thesis=_THESIS,
        symbol_character=_CHARACTER,
        profile=_PROFILE,
        window=_FETCH_WINDOW,
        mandate=mandate,
        context_window=_WINDOW,
    )


def test_a_renderer_reflects_a_disk_change_because_it_calls_the_builder_that_reads_disk(
    tmp_path,
) -> None:
    box, ledger, mandate = _populate(tmp_path)
    site_input = FormulateSiteInput(
        toolbox=box, ledger=ledger, context_window=_WINDOW, mandate=mandate
    )
    before = FORMULATE_SITE.render(site_input, HarnessSpec())

    ledger.record_thesis("newidea", "NEWDISKSENTINEL breakout on opening gaps")
    after = FORMULATE_SITE.render(site_input, HarnessSpec())

    assert "NEWDISKSENTINEL" not in before
    assert "NEWDISKSENTINEL" in after


def test_an_ablated_harness_spec_does_not_yet_change_what_a_renderer_forwards(tmp_path) -> None:
    """Ablations are wired by a later epic; today the adapter forwards the builder's bytes only."""
    box, ledger, mandate = _populate(tmp_path)
    site_input = FormulateSiteInput(
        toolbox=box, ledger=ledger, context_window=_WINDOW, mandate=mandate
    )

    production = FORMULATE_SITE.render(site_input, HarnessSpec())
    ablated = FORMULATE_SITE.render(
        site_input, HarnessSpec(contract_sheet=False, worked_example=None, retry_hints=False)
    )

    assert ablated == production


def test_a_renderer_honours_the_context_window_its_input_declares(tmp_path) -> None:
    """The builder's own fit assertion is reached unchanged: a smaller window trims advisories."""
    box, ledger, mandate = _populate(tmp_path)
    wide = FORMULATE_SITE.render(
        FormulateSiteInput(toolbox=box, ledger=ledger, context_window=_WINDOW, mandate=mandate),
        HarnessSpec(),
    )

    narrow = FORMULATE_SITE.render(
        FormulateSiteInput(toolbox=box, ledger=ledger, context_window=1200, mandate=mandate),
        HarnessSpec(),
    )

    assert len(narrow) < len(wide)
    assert narrow == formulate_briefing(box, ledger, mandate=mandate, context_window=1200)


# ── the knob sets name existing settings knobs, and promote nothing ──────────────────────────
def test_every_knob_the_episodic_sites_declare_is_a_field_of_the_research_agent_settings() -> None:
    settings_knobs = set(AgentResearchConfig.model_fields)

    for declared in EPISODIC_SITES:
        assert declared.knobs.knob_names() <= settings_knobs, declared.id


def test_each_episodic_site_declares_the_session_model_the_token_levers_and_the_retry_count() -> (
    None
):
    expected = frozenset({"model", "max_tokens", "context_window", "episode_retries"})

    assert FormulateKnobs.knob_names() == expected
    assert DecideKnobs.knob_names() == expected
    assert DiscoverKnobs.knob_names() == expected


def test_each_episodic_site_declares_its_own_knob_type() -> None:
    """Per-site knob sets: two sites sharing one type could not diverge without breaking both."""
    types = [declared.knobs for declared in EPISODIC_SITES]

    assert types == [FormulateKnobs, DecideKnobs, DiscoverKnobs]
    assert all(issubclass(knobs, SiteKnobs) for knobs in types)


def test_an_override_naming_a_knob_no_episodic_site_has_is_refused_before_it_is_spent() -> None:
    from noctis.eval.knobs import UnknownKnob

    with pytest.raises(UnknownKnob) as error:
        HarnessSpec(knob_overrides={"temperature": 0.2}).validate_knobs(DecideKnobs)

    assert "temperature" in str(error.value)
