"""The failure-taxonomy mechanism (#199): per-site classifiers with a reserved catch-all.

The mechanism only — no site's vocabulary lives here. What is asserted is the contract every
vocabulary is held to: a closed set of named classes with pure matchers, registered per site into
an instance a caller owns; a batch classified into counts and shares that do not depend on the
order the errors arrived in; a catch-all class that is always reported, because its share rising is
how anyone learns the classifiers have rotted; and refusals loud enough that a bad vocabulary dies
at registration rather than in a benchmark record nobody can re-read.
"""

from __future__ import annotations

import ast
import random
from pathlib import Path

import pytest

import noctis.eval.taxonomy as taxonomy_module
from noctis.eval.taxonomy import (
    UNCLASSIFIED,
    FailureClass,
    FailureTaxonomy,
    UnknownVocabulary,
    VocabularyError,
)

TAXONOMY_SOURCE = Path(taxonomy_module.__file__)


def _syntax(error: str) -> bool:
    return "SyntaxError" in error


def _timeout(error: str) -> bool:
    return "timed out" in error


def _lookahead(error: str) -> bool:
    return "lookahead" in error


def _widget_vocabulary() -> tuple[FailureClass, ...]:
    return (
        FailureClass(name="syntax", matches=_syntax),
        FailureClass(name="timeout", matches=_timeout),
    )


def _taxonomy_with_one_site(site_id: str = "widget") -> FailureTaxonomy:
    taxonomy = FailureTaxonomy()
    taxonomy.register(site_id, _widget_vocabulary())
    return taxonomy


# ── a site registers a closed vocabulary ──────────────────────────────────────────────────
def test_a_site_registers_a_closed_vocabulary_of_failure_classes() -> None:
    taxonomy = _taxonomy_with_one_site()

    assert taxonomy.registered_sites() == ("widget",)
    assert "syntax" in taxonomy.class_names("widget")
    assert "timeout" in taxonomy.class_names("widget")


def test_a_sites_classes_are_listed_in_registration_order_with_the_catch_all_last() -> None:
    taxonomy = _taxonomy_with_one_site()

    assert taxonomy.class_names("widget") == ("syntax", "timeout", UNCLASSIFIED)


def test_one_error_classifies_to_the_name_of_the_class_whose_matcher_recognises_it() -> None:
    taxonomy = _taxonomy_with_one_site()

    assert taxonomy.classify_one("widget", "SyntaxError: unexpected EOF") == "syntax"
    assert taxonomy.classify_one("widget", "the subprocess timed out after 30s") == "timeout"


# ── counts and shares over a batch ────────────────────────────────────────────────────────
def test_classifying_a_batch_reports_a_count_for_every_class_in_the_vocabulary() -> None:
    taxonomy = _taxonomy_with_one_site()

    breakdown = taxonomy.classify(
        "widget",
        ["SyntaxError: bad", "SyntaxError: worse", "timed out", "the gate said no"],
    )

    assert dict(breakdown.counts) == {"syntax": 2, "timeout": 1, UNCLASSIFIED: 1}
    assert breakdown.total == 4


def test_classifying_a_batch_reports_each_classes_share_of_the_batch() -> None:
    taxonomy = _taxonomy_with_one_site()

    breakdown = taxonomy.classify(
        "widget",
        ["SyntaxError: bad", "SyntaxError: worse", "timed out", "the gate said no"],
    )

    assert dict(breakdown.shares) == {"syntax": 0.5, "timeout": 0.25, UNCLASSIFIED: 0.25}


def test_the_shares_of_a_classified_batch_sum_to_one() -> None:
    taxonomy = _taxonomy_with_one_site()

    breakdown = taxonomy.classify("widget", ["SyntaxError", "timed out", "who knows"])

    assert sum(breakdown.shares.values()) == pytest.approx(1.0)


def test_a_shuffled_batch_classifies_to_the_same_counts_and_shares() -> None:
    errors = [
        "SyntaxError: bad",
        "timed out",
        "the gate said no",
        "SyntaxError: worse",
        "timed out again",
    ]
    taxonomy = _taxonomy_with_one_site()
    shuffled = list(errors)
    random.Random(20260801).shuffle(shuffled)

    assert shuffled != errors
    assert taxonomy.classify("widget", shuffled) == taxonomy.classify("widget", errors)


def test_an_empty_batch_reports_every_class_at_zero() -> None:
    taxonomy = _taxonomy_with_one_site()

    breakdown = taxonomy.classify("widget", [])

    assert breakdown.total == 0
    assert dict(breakdown.counts) == {"syntax": 0, "timeout": 0, UNCLASSIFIED: 0}
    assert dict(breakdown.shares) == {"syntax": 0.0, "timeout": 0.0, UNCLASSIFIED: 0.0}


# ── the catch-all ─────────────────────────────────────────────────────────────────────────
def test_an_error_no_matcher_recognises_lands_in_the_catch_all_class() -> None:
    taxonomy = _taxonomy_with_one_site()

    assert taxonomy.classify_one("widget", "something nobody wrote a matcher for") == UNCLASSIFIED


def test_the_catch_all_class_is_reported_even_when_every_error_was_classified() -> None:
    taxonomy = _taxonomy_with_one_site()

    breakdown = taxonomy.classify("widget", ["SyntaxError: bad", "timed out"])

    assert breakdown.counts[UNCLASSIFIED] == 0
    assert breakdown.shares[UNCLASSIFIED] == 0.0


def test_the_catch_all_share_is_readable_on_its_own_as_the_rot_signal() -> None:
    taxonomy = _taxonomy_with_one_site()

    breakdown = taxonomy.classify("widget", ["SyntaxError: bad", "who knows", "nor me", "nor I"])

    assert breakdown.unclassified_share == 0.75


# ── overlap is decided, not accidental ────────────────────────────────────────────────────
def test_the_first_registered_class_wins_when_two_matchers_match_one_error() -> None:
    taxonomy = FailureTaxonomy()
    taxonomy.register(
        "widget",
        (
            FailureClass(name="syntax", matches=_syntax),
            FailureClass(name="lookahead", matches=_lookahead),
        ),
    )

    assert taxonomy.classify_one("widget", "SyntaxError near the lookahead guard") == "syntax"


def test_registration_order_decides_the_winner_of_an_overlap() -> None:
    taxonomy = FailureTaxonomy()
    taxonomy.register(
        "widget",
        (
            FailureClass(name="lookahead", matches=_lookahead),
            FailureClass(name="syntax", matches=_syntax),
        ),
    )

    assert taxonomy.classify_one("widget", "SyntaxError near the lookahead guard") == "lookahead"


# ── vocabularies are strictly per-site ────────────────────────────────────────────────────
def test_two_sites_may_register_the_same_class_name_without_colliding() -> None:
    taxonomy = FailureTaxonomy()
    taxonomy.register("widget", (FailureClass(name="refused", matches=_syntax),))
    taxonomy.register("gadget", (FailureClass(name="refused", matches=_timeout),))

    assert taxonomy.classify_one("widget", "SyntaxError: bad") == "refused"
    assert taxonomy.classify_one("gadget", "SyntaxError: bad") == UNCLASSIFIED
    assert taxonomy.classify_one("gadget", "timed out") == "refused"


def test_one_sites_classes_never_appear_in_another_sites_breakdown() -> None:
    taxonomy = _taxonomy_with_one_site()
    taxonomy.register("gadget", (FailureClass(name="lookahead", matches=_lookahead),))

    breakdown = taxonomy.classify("gadget", ["SyntaxError: bad", "lookahead in on_bar"])

    assert dict(breakdown.counts) == {"lookahead": 1, UNCLASSIFIED: 1}


def test_a_sites_classification_never_consults_another_sites_matchers() -> None:
    taxonomy = _taxonomy_with_one_site()
    taxonomy.register("gadget", (FailureClass(name="lookahead", matches=_lookahead),))

    assert taxonomy.classify_one("widget", "lookahead in on_bar") == UNCLASSIFIED


def test_two_taxonomies_do_not_observe_each_others_registrations() -> None:
    """The house posture: a registry is an instance a caller constructs, never a module global."""
    mine = _taxonomy_with_one_site()
    yours = FailureTaxonomy()

    assert yours.registered_sites() == ()
    assert mine.registered_sites() == ("widget",)


# ── refusals are loud ─────────────────────────────────────────────────────────────────────
def test_registering_a_duplicate_class_name_within_a_site_is_refused() -> None:
    taxonomy = FailureTaxonomy()

    with pytest.raises(VocabularyError) as error:
        taxonomy.register(
            "widget",
            (
                FailureClass(name="syntax", matches=_syntax),
                FailureClass(name="syntax", matches=_timeout),
            ),
        )

    assert "syntax" in str(error.value)
    assert "widget" in str(error.value)


def test_registering_the_reserved_catch_all_name_is_refused() -> None:
    taxonomy = FailureTaxonomy()

    with pytest.raises(VocabularyError) as error:
        taxonomy.register("widget", (FailureClass(name=UNCLASSIFIED, matches=_syntax),))

    assert UNCLASSIFIED in str(error.value)


def test_a_refused_registration_leaves_the_site_unregistered() -> None:
    taxonomy = FailureTaxonomy()

    with pytest.raises(VocabularyError):
        taxonomy.register("widget", (FailureClass(name=UNCLASSIFIED, matches=_syntax),))

    assert taxonomy.registered_sites() == ()


def test_registering_a_second_vocabulary_for_one_site_is_refused() -> None:
    """A vocabulary is closed: extending one later would silently re-label yesterday's records."""
    taxonomy = _taxonomy_with_one_site()

    with pytest.raises(VocabularyError) as error:
        taxonomy.register("widget", (FailureClass(name="lookahead", matches=_lookahead),))

    assert "widget" in str(error.value)


def test_registering_a_vocabulary_with_no_classes_is_refused() -> None:
    taxonomy = FailureTaxonomy()

    with pytest.raises(VocabularyError) as error:
        taxonomy.register("widget", ())

    assert "widget" in str(error.value)


def test_classifying_against_a_site_with_no_registered_vocabulary_is_refused() -> None:
    taxonomy = _taxonomy_with_one_site()

    with pytest.raises(UnknownVocabulary) as error:
        taxonomy.classify("gadget", ["SyntaxError: bad"])

    assert "gadget" in str(error.value)


def test_the_refusal_for_an_unregistered_site_names_the_sites_that_are_registered() -> None:
    taxonomy = _taxonomy_with_one_site()

    with pytest.raises(UnknownVocabulary) as error:
        taxonomy.classify_one("gadget", "SyntaxError: bad")

    assert "widget" in str(error.value)


def test_the_refusal_on_an_empty_taxonomy_says_no_site_is_registered() -> None:
    taxonomy = FailureTaxonomy()

    with pytest.raises(UnknownVocabulary) as error:
        taxonomy.class_names("widget")

    assert "no sites" in str(error.value)


# ── what shared reporting sees ────────────────────────────────────────────────────────────
def test_a_breakdown_carries_class_names_and_counts_and_nothing_of_the_site() -> None:
    """Shared reporting is per-site-blind by construction: no site id, no matcher, in the record."""
    breakdown = _taxonomy_with_one_site().classify("widget", ["SyntaxError: bad"])

    assert {field for field in vars(breakdown)} == {"counts", "shares", "total"}


def test_the_counts_of_a_breakdown_cannot_be_mutated_by_a_reader() -> None:
    breakdown = _taxonomy_with_one_site().classify("widget", ["SyntaxError: bad"])

    with pytest.raises(TypeError):
        breakdown.counts["syntax"] = 99  # type: ignore[index]


def test_the_vocabulary_a_caller_passed_cannot_be_changed_under_the_taxonomy() -> None:
    classes = [FailureClass(name="syntax", matches=_syntax)]
    taxonomy = FailureTaxonomy()
    taxonomy.register("widget", classes)

    classes.append(FailureClass(name="timeout", matches=_timeout))

    assert taxonomy.class_names("widget") == ("syntax", UNCLASSIFIED)


# ── purity, structurally ──────────────────────────────────────────────────────────────────
def _imports(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


def test_the_taxonomy_module_reaches_no_io_no_clock_and_no_randomness() -> None:
    """Classification is a pure function of the strings it is handed — the same rule
    ``run_record`` and ``metrics`` are held to. The import allowlist is the structural half
    (nothing here can reach a file, a clock or the settings even transitively); the text check
    is the half that catches a clock or a file opened through a name that *is* allowed."""
    assert _imports(TAXONOMY_SOURCE) <= {
        "__future__",
        "collections",
        "dataclasses",
        "types",
        "typing",
    }
    text = TAXONOMY_SOURCE.read_text(encoding="utf-8")
    for forbidden in ("datetime", "utcnow", "open(", "Path(", "os.", "random.", "time.", "json."):
        assert forbidden not in text, forbidden


def _module_level_bindings(source: Path) -> dict[str, ast.expr]:
    """Every ``name = value`` at module level, by bound name (dunders excluded)."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    bound: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            bound[node.target.id] = node.value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound[target.id] = node.value
    return {name: value for name, value in bound.items() if not name.startswith("__")}


def test_the_taxonomy_module_binds_no_module_level_mutable_registry() -> None:
    """No global to reset: a taxonomy is an instance, so two benchmarks cannot leak into one."""
    containers = sorted(
        name
        for name, value in _module_level_bindings(TAXONOMY_SOURCE).items()
        if isinstance(value, ast.Dict | ast.List | ast.Set | ast.DictComp | ast.ListComp)
    )

    assert not containers, f"module-level mutable state: {containers}"


def test_the_module_level_binding_walk_sees_the_modules_own_constants() -> None:
    """Non-vacuity: the check above only means something if the walk reaches real bindings."""
    assert {"UNCLASSIFIED", "Matcher"} <= set(_module_level_bindings(TAXONOMY_SOURCE))
