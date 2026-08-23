"""Boundary closure (#194): the site set is pinned, its absences are stated, every site resolves.

Three jobs, and they are deliberately in one file because they are one question — *is the registry
complete and honest?*

The **pin** is a ratchet on the site set itself: a sixth declaration, or a renamed id, fails here
until someone updates the literal below, which is the conscious act the epic wants. The
**exclusion** tests hold the registry to documenting what it deliberately leaves out, asserted on
key phrases the way ``tests/test_eval_boundary.py`` holds the package docstring to stating the
one-way rule. The **sweep** resolves every declared site against the production objects it claims
to bind — the driver's own contract constants by identity, a renderer importable from the module it
says it lives in, a knob type in the shared hierarchy — so a renamed briefing builder or a copied
contract breaks the build rather than the first benchmark run.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from noctis.eval.coder_case import CODER_AXES, Axis
from noctis.eval.coder_scorer import CODER_SCORER
from noctis.eval.decide_site import DECIDE_SCORER, DIFFICULTY_AXES
from noctis.eval.knobs import SiteKnobs
from noctis.eval.registry import DECLARED_SITES, SITES, site, sites
from noctis.research.driver import DECIDE_CONTRACT, DISCOVER_CONTRACT, FORMULATE_CONTRACT
from noctis.research.episode import EmitContract

# The pin. Editing this set is how a site is added or renamed — deliberately, in a reviewable diff.
DECLARED_SITE_IDS = frozenset({"coder", "formulate", "decide", "discover", "distill"})

# What each declared site's ``contract`` must *be*, by identity: the episodic driver's own emit
# contract objects, or ``None`` for the two sites production types no output for (the coder, judged
# by the write gate; the distiller, judged by nothing but its bullet filter).
EXPECTED_CONTRACTS: dict[str, EmitContract[Any] | None] = {
    "coder": None,
    "formulate": FORMULATE_CONTRACT,
    "decide": DECIDE_CONTRACT,
    "discover": DISCOVER_CONTRACT,
    "distill": None,
}

# What each declared site's ``scorers`` slot must carry, by identity — a site absent from here
# declares none. Filling a slot is the deliberate edit adding a scorer takes: a site that starts
# scoring itself without a line here is a benchmark growing a judge nobody reviewed.
EXPECTED_SCORERS: dict[str, tuple[Any, ...]] = {
    "coder": (CODER_SCORER,),
    "decide": (DECIDE_SCORER,),
}

# What each declared site's ``difficulty_axes`` slot must carry: the axis vocabulary its corpus
# labels cases on, and the tuple the runner hands its scorers to stratify a reading by. A site
# absent from here declares none — cases nobody labelled have no axes, which is an absence and not
# an empty measurement.
EXPECTED_AXES: dict[str, tuple[str, ...]] = {
    "coder": CODER_AXES,
    "decide": DIFFICULTY_AXES,
}


def _registry_docstring() -> str:
    """The registry module's own prose, whitespace-collapsed and lowered for phrase assertions."""
    module = importlib.import_module("noctis.eval.registry")
    return " ".join((module.__doc__ or "").lower().split())


def _resolve(dotted_module: str, qualname: str) -> Any:
    """Import ``dotted_module`` and walk ``qualname`` — what "importable" means for a renderer."""
    resolved: Any = importlib.import_module(dotted_module)
    for part in qualname.split("."):
        resolved = getattr(resolved, part)
    return resolved


# ── the pin: exactly five sites, and changing that is a conscious act ─────────────────────
def test_the_registry_declares_exactly_the_five_benchmarkable_sites() -> None:
    assert set(SITES) == DECLARED_SITE_IDS


def test_the_declaration_tuple_and_the_index_name_the_same_five_sites() -> None:
    assert [declaration.id for declaration in DECLARED_SITES] == list(SITES)
    assert len(DECLARED_SITES) == len(DECLARED_SITE_IDS)


# ── the absences, stated at the registry ──────────────────────────────────────────────────
def test_the_registry_documents_the_conversation_loop_as_deliberately_undeclared() -> None:
    docstring = _registry_docstring()

    assert "conversation loop" in docstring
    assert "not a site" in docstring
    assert "transcript" in docstring
    assert "parity harness" in docstring


def test_the_registry_documents_onboarding_verify_as_deliberately_undeclared() -> None:
    docstring = _registry_docstring()

    assert "onboarding-verify" in docstring
    assert "liveness check" in docstring


# ── the sweep: every declared site resolves end to end, in one loop ───────────────────────
def test_every_declared_site_resolves_end_to_end_against_the_objects_it_binds() -> None:
    """One loop over the whole registry — the net a renamed builder or a drifted copy falls into."""
    seen: set[str] = set()

    for declaration in sites():
        site_id = declaration.id

        # The id is this site's own, unique across the registry, and looks itself back up.
        assert site_id not in seen, f"{site_id} is declared twice"
        seen.add(site_id)
        assert site(site_id) is declaration

        # The declared generation a record refuses a cross-generation comparison on.
        assert isinstance(declaration.version, str) and declaration.version.strip(), site_id

        # The contract is the driver's own object, by identity — or honestly absent.
        assert site_id in EXPECTED_CONTRACTS, f"{site_id} has no expected contract pinned"
        assert declaration.contract is EXPECTED_CONTRACTS[site_id], site_id

        # The renderer is importable from the module it says it lives in, and takes (input, spec).
        render = declaration.render
        assert callable(render), site_id
        assert _resolve(render.__module__, render.__qualname__) is render, site_id
        assert len(inspect.signature(render).parameters) == 2, site_id

        # The knob type is a real SiteKnobs subclass a harness can validate an override against.
        assert isinstance(declaration.knobs, type), site_id
        assert issubclass(declaration.knobs, SiteKnobs), site_id
        assert declaration.knobs is not SiteKnobs, site_id

        # Scorers arrive per site, with the epic that can honestly score one. Decide's is the
        # agreement scorer (#209) and the coder's is its co-primary pass-rate reading (#226);
        # every other slot is still honestly empty.
        assert declaration.scorers == EXPECTED_SCORERS.get(site_id, ()), site_id

        # The difficulty axes its corpus labels cases on — declared beside its scorers, so one
        # site-neutral loop stratifies either site's reading without importing either's
        # vocabulary (#306).
        assert declaration.difficulty_axes == EXPECTED_AXES.get(site_id, ()), site_id

    assert seen == DECLARED_SITE_IDS


# ── the axes themselves: what the two scoring sites stratify their readings by ────────────
def test_the_coder_declares_the_seven_axes_a_labelled_brief_carries() -> None:
    """The enum is the vocabulary; the declaration publishes it flat, as a record's keys."""
    assert site("coder").difficulty_axes == tuple(axis.value for axis in Axis)
    assert len(site("coder").difficulty_axes) == 7


def test_decide_declares_the_three_axes_a_mined_case_carries() -> None:
    assert site("decide").difficulty_axes == ("margin", "binding_gate", "evidence_depth")


def test_a_site_whose_cases_carry_no_difficulty_labels_declares_no_axes() -> None:
    assert [declared.id for declared in sites() if not declared.difficulty_axes] == [
        "formulate",
        "discover",
        "distill",
    ]
