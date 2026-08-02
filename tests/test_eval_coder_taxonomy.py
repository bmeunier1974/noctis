"""The coder's closed failure vocabulary (#215): ten classes over real write-gate wording.

Every fixture string here is the *production* wording a coder failure actually wears — quoted from
the module and line that emits it, or (better) obtained by driving the engine that emits it — so a
gate that rewords itself fails a test here instead of quietly emptying a class into the escape
hatch. Assertions are external behaviour only: the class name a string classifies to, the knob a
class annotates, what a registered taxonomy reports for a batch.
"""

from __future__ import annotations

import ast
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

import noctis.eval.coder_taxonomy as coder_taxonomy_module
from noctis.eval.coder_distill_sites import CODER_SITE
from noctis.eval.coder_taxonomy import (
    CODER_CLASSES,
    CODER_SITE_ID,
    OTHER,
    classify_coder_failure,
    coder_failure_classes,
    knob_for,
    register_coder_taxonomy,
)
from noctis.eval.taxonomy import UNCLASSIFIED, FailureTaxonomy
from noctis.research import author, contract_sheet
from noctis.strategies.library import WARMUP_TOO_LARGE_MARKER, is_warmup_too_large

MODULE_SOURCE = Path(coder_taxonomy_module.__file__)


def _registered() -> FailureTaxonomy:
    taxonomy = FailureTaxonomy()
    register_coder_taxonomy(taxonomy)
    return taxonomy


def _class_named(name: str):
    return next(declared for declared in CODER_CLASSES if declared.name == name)


# ── one real-shaped fixture per class ─────────────────────────────────────────────────────
def test_a_prose_only_reply_classifies_as_no_code_block() -> None:
    # The production correction itself (noctis.research.author._extraction_error, non-length stop).
    error = str(author._extraction_error("end_turn"))

    assert classify_coder_failure(error) == "no_code_block"


def test_a_reply_cut_off_by_the_output_token_limit_classifies_as_truncated() -> None:
    # The production correction for stop_reason == "length" (the same author engine).
    error = str(author._extraction_error("length"))

    assert classify_coder_failure(error) == "truncated"


def test_a_missing_third_party_module_classifies_as_import_error() -> None:
    # The subprocess validator prints "{type(exc).__name__}: {exc}" and the gate raises that last
    # line verbatim (src/noctis/strategies/library.py::_main, ::validate_in_subprocess).
    error = "ModuleNotFoundError: No module named 'talib'"

    assert classify_coder_failure(error) == "import_error"


def test_a_source_file_that_does_not_parse_classifies_as_import_error() -> None:
    error = "SyntaxError: invalid syntax (my_strategy.py, line 42)"

    assert classify_coder_failure(error) == "import_error"


def test_a_module_holding_no_single_strategy_class_classifies_as_import_error() -> None:
    # src/noctis/strategies/library.py::_find_strategy_class — the file imported but never yielded
    # the one strategy class the gate could go on to grade.
    error = "StrategyValidationError: expected exactly one TraderStrategy subclass, found 0"

    assert classify_coder_failure(error) == "import_error"


def test_an_unexpected_keyword_argument_classifies_as_api_misuse() -> None:
    # tests/test_contract_sheet.py's own hint fixture — the shape the enricher recognises.
    error = "ExitRules.__init__() got an unexpected keyword argument 'target_pct'"

    assert classify_coder_failure(error) == "api_misuse"


def test_a_state_update_arity_mistake_classifies_as_api_misuse() -> None:
    error = "TypeError: AtrState.update() takes 2 positional arguments but 4 were given"

    assert classify_coder_failure(error) == "api_misuse"


def test_a_class_name_disagreeing_with_the_file_name_classifies_as_name_mismatch() -> None:
    # src/noctis/strategies/library.py::_validate_file.
    error = (
        "StrategyValidationError: class sets name='mismatch' but the strategy/file name is 'probe'"
    )

    assert classify_coder_failure(error) == "name_mismatch"


def test_the_startup_name_mismatch_wording_also_classifies_as_name_mismatch() -> None:
    # src/noctis/strategies/library.py::load_and_register — the same defect, the other wording.
    error = "class name attribute 'Probe' != file name 'probe'"

    assert classify_coder_failure(error) == "name_mismatch"


def test_an_unmet_scenario_expectation_classifies_as_scenario_violation() -> None:
    # src/noctis/strategies/scenarios.py::run_scenario — expectation failure + observed behaviour.
    error = (
        "StrategyValidationError: scenario 'breakout': never long in [80, 120] — observed: never "
        "took a position across all 240 bars — dead or late logic: no long entry where the thesis "
        "demands one; check warmup length and the entry condition's direction"
    )

    assert classify_coder_failure(error) == "scenario_violation"


def test_a_tape_shape_refusal_classifies_as_scenario_violation() -> None:
    # src/noctis/strategies/scenarios.py::check_scenario_contract.
    error = "StrategyValidationError: at least one scenario must be a no-trade tape (always_flat())"

    assert classify_coder_failure(error) == "scenario_violation"


def test_a_hand_written_oracle_on_the_spec_path_classifies_as_scenario_violation() -> None:
    # src/noctis/strategies/library.py::_validate_against_spec — the stamped tape is the oracle.
    error = (
        "StrategyValidationError: spec-driven write: the known-outcome oracle is fixed by the "
        "supplied scenario spec and is machine-stamped by the gate — remove the scenarios() method "
        "from the source and change only the trading logic (on_start/on_bar/param_space) to "
        "satisfy the fixed oracle"
    )

    assert classify_coder_failure(error) == "scenario_violation"


def test_a_warmup_too_large_for_the_fixed_oracle_classifies_as_warmup_too_large() -> None:
    # src/noctis/strategies/library.py::_validate_against_spec, around WARMUP_TOO_LARGE_MARKER.
    error = (
        f"StrategyValidationError: declared warmup_bars=400 is {WARMUP_TOO_LARGE_MARKER}: tape "
        "'chop' is 240 bars. Shrink the lookback defaults in Params so the strategy warms up "
        "faster — never enlarge the scenario tape to fit the warmup"
    )

    assert classify_coder_failure(error) == "warmup_too_large"


def test_a_vectorised_override_that_looks_ahead_classifies_as_structural_invariant() -> None:
    # src/noctis/strategies/scenarios.py::_check_truncation.
    error = (
        "StrategyValidationError: scenario 'chop': signals() looks ahead — signals(tape[:60]) "
        "disagrees with signals(tape)[:60] at bar 41 (1 vs 0); a vectorised override must decide "
        "bar t from bars ≤ t only (no full-series max/mean, no centered window, no shift(-k))"
    )

    assert classify_coder_failure(error) == "structural_invariant"


def test_a_price_scale_dependent_strategy_classifies_as_structural_invariant() -> None:
    # src/noctis/strategies/scenarios.py::_check_price_scale.
    error = (
        "StrategyValidationError: scenario 'trendy': price-scale dependent — scaling every price "
        "×10 flips the target at bar 73 (1 vs 0); decide from scale-free features (moving "
        "averages, ratios, percentiles), never absolute price levels, so the edge transfers "
        "across a symbol panel"
    )

    assert classify_coder_failure(error) == "structural_invariant"


def test_a_non_deterministic_replay_classifies_as_structural_invariant() -> None:
    # src/noctis/strategies/scenarios.py::_check_determinism.
    error = (
        "StrategyValidationError: scenario 'chop': non-deterministic replay — targets differ at "
        "bar 88 (1 vs 0) between two replays of the same tape; remove class-level mutable state "
        "or randomness and reset all incremental state in on_start (on_bar must be a pure "
        "function of the bars seen so far)"
    )

    assert classify_coder_failure(error) == "structural_invariant"


def test_a_dishonest_warmup_classifies_as_structural_invariant() -> None:
    # src/noctis/strategies/scenarios.py::_check_warmup_honesty.
    error = (
        "StrategyValidationError: scenario 'trendy': warmup dishonest — took a long position at "
        "bar 4, before the declared warmup_bars=20 (promised flat through bar 19); raise "
        "warmup_bars or delay the entry"
    )

    assert classify_coder_failure(error) == "structural_invariant"


def test_a_source_level_structural_defect_classifies_as_structural_invariant() -> None:
    # src/noctis/strategies/structure.py::_check_duplicate_definitions (the raw-source lint).
    error = (
        "StrategyValidationError: duplicate definition of 'on_bar' in class 'Probe' — the second "
        "silently replaces the first"
    )

    assert classify_coder_failure(error) == "structural_invariant"


def test_a_vectorised_override_disagreeing_with_the_per_bar_path_classifies_as_signals_parity() -> (
    None
):
    # src/noctis/strategies/library.py::_validate_file.
    error = (
        "StrategyValidationError: signals() disagrees with the on_bar replay on the fixture "
        "(parity violation); drop the signals() override or fix it"
    )

    assert classify_coder_failure(error) == "signals_parity"


def test_an_error_no_matcher_recognises_classifies_as_the_escape_hatch() -> None:
    assert classify_coder_failure("the coder replied in Latin") == OTHER


# ── the escape hatch never raises and never guesses ───────────────────────────────────────
def test_the_escape_hatch_is_the_mechanisms_reserved_catch_all() -> None:
    """One bucket, not two: the epic's ``other`` IS the mechanism's reserved catch-all class."""
    assert OTHER == UNCLASSIFIED


def test_an_empty_error_string_lands_in_the_escape_hatch() -> None:
    assert classify_coder_failure("") == OTHER


@given(st.text())
def test_classifying_arbitrary_text_never_raises_and_always_names_a_declared_class(
    error: str,
) -> None:
    assert classify_coder_failure(error) in {declared.name for declared in CODER_CLASSES}


# ── precedence over multi-signal strings ──────────────────────────────────────────────────
def test_a_warmup_too_large_refusal_naming_a_tape_beats_the_scenario_match() -> None:
    error = (
        f"scenario 'chop': declared warmup_bars=400 is {WARMUP_TOO_LARGE_MARKER}: 240 bars is "
        "too short"
    )

    assert classify_coder_failure(error) == "warmup_too_large"


def test_a_truncated_reply_beats_the_no_code_block_match_it_also_mentions() -> None:
    """Both corrections talk about a missing code block; only one names the real cause."""
    error = str(author._extraction_error("length"))

    assert "code block" in error
    assert classify_coder_failure(error) == "truncated"


def test_a_helper_kwarg_mistake_wrapped_by_the_scenario_declaration_stays_api_misuse() -> None:
    # tests/test_contract_sheet.py's wrapped fixture: the enricher resolves it, so must the class.
    error = "scenarios() raised TypeError: trend() got an unexpected keyword argument 'drift'"

    assert classify_coder_failure(error) == "api_misuse"


def test_a_structural_invariant_on_a_named_tape_beats_the_scenario_violation_match() -> None:
    """Tier-1 invariant failures wear the ``scenario '<name>':`` prefix too — and outrank it."""
    error = "scenario 'chop': price-scale dependent — scaling every price ×10 flips the target"

    assert classify_coder_failure(error) == "structural_invariant"


def test_a_replay_type_error_naming_a_declared_helper_beats_the_import_match() -> None:
    """An unwrapped ``TypeError:`` head reads as import failure — unless the API shape names it."""
    error = "TypeError: ExitRules.__init__() got an unexpected keyword argument 'target_pct'"

    assert classify_coder_failure(error) == "api_misuse"


def test_the_registration_order_is_the_declared_precedence_order() -> None:
    assert [declared.name for declared in CODER_CLASSES] == [
        "warmup_too_large",
        "api_misuse",
        "name_mismatch",
        "signals_parity",
        "structural_invariant",
        "scenario_violation",
        "truncated",
        "no_code_block",
        "import_error",
        OTHER,
    ]


# ── the recognizers are the engine's own, not copies ──────────────────────────────────────
def test_the_warmup_class_recognises_through_the_engines_shared_predicate() -> None:
    """The drift guard: reword the gate's marker and this class stops matching, loudly."""
    assert _class_named("warmup_too_large").matches is is_warmup_too_large


def test_the_warmup_class_matches_exactly_what_the_engine_marker_matches() -> None:
    error = f"declared warmup_bars=400 is {WARMUP_TOO_LARGE_MARKER}: no room"

    assert is_warmup_too_large(error)
    assert classify_coder_failure(error) == "warmup_too_large"
    assert not is_warmup_too_large("a warmup that is merely large")
    assert classify_coder_failure("a warmup that is merely large") != "warmup_too_large"


def test_the_api_misuse_class_reads_the_hint_matchers_own_patterns() -> None:
    """No second copy of the two shapes: the same compiled objects the enricher searches with."""
    assert coder_taxonomy_module.UPDATE_ARITY_PATTERN is contract_sheet._UPDATE_ARITY_RE
    assert coder_taxonomy_module.UNEXPECTED_KWARG_PATTERN is contract_sheet._UNEXPECTED_KWARG_RE


def test_every_error_the_hint_enricher_recognises_classifies_as_api_misuse() -> None:
    """Whatever the retry prompt can enrich with a true signature is, by definition, API misuse."""
    enriched = [
        "AtrState.update() takes 2 positional arguments but 4 were given",
        "AtrState.update() missing 1 required positional argument: 'bar'",
        "ExitRules.__init__() got an unexpected keyword argument 'target_pct'",
        "scenarios() raised TypeError: trend() got an unexpected keyword argument 'drift'",
        "scenarios() raised TypeError: long_within() got an unexpected keyword argument 'bad'",
    ]

    for error in enriched:
        assert contract_sheet.hint_for_gate_error(error) is not None, error
        assert classify_coder_failure(error) == "api_misuse", error


# ── the knob each class points at ─────────────────────────────────────────────────────────
def test_the_vocabulary_is_ten_classes() -> None:
    assert len(CODER_CLASSES) == 10


def test_every_class_carries_the_knob_its_share_points_at() -> None:
    assert knob_for("truncated") == "coder max-tokens, thinking allowance, thinking dial"
    assert knob_for("api_misuse") == "contract sheet coverage, hint enrichment"
    assert knob_for("structural_invariant") == "model capability — the serious one"
    assert all(declared.knob for declared in CODER_CLASSES)


def test_the_escape_hatchs_knob_is_to_grow_the_taxonomy() -> None:
    assert knob_for(OTHER) == "grow the taxonomy"


def test_asking_for_the_knob_of_a_class_the_vocabulary_does_not_declare_is_refused() -> None:
    try:
        knob_for("gremlins")
    except KeyError as error:
        assert "gremlins" in str(error)
    else:  # pragma: no cover - the assertion below is the failure report
        raise AssertionError("an undeclared class name must be refused, not answered")


# ── registration into the mechanism ───────────────────────────────────────────────────────
def test_the_vocabulary_registers_against_the_declared_coder_sites_id() -> None:
    assert CODER_SITE_ID == CODER_SITE.id == "coder"


def test_registering_declares_the_ten_class_names_for_the_coder_site() -> None:
    taxonomy = _registered()

    assert taxonomy.registered_sites() == ("coder",)
    assert taxonomy.class_names("coder") == tuple(declared.name for declared in CODER_CLASSES)


def test_the_registered_classes_exclude_the_reserved_catch_all_the_mechanism_owns() -> None:
    """Nine matchers are registered; the tenth class is the mechanism's own catch-all."""
    assert [declared.name for declared in coder_failure_classes()] == [
        declared.name for declared in CODER_CLASSES if declared.name != OTHER
    ]


def test_a_registered_taxonomy_counts_a_batch_of_coder_failures_by_class() -> None:
    taxonomy = _registered()

    breakdown = taxonomy.classify(
        "coder",
        [
            "ModuleNotFoundError: No module named 'talib'",
            "ExitRules.__init__() got an unexpected keyword argument 'target_pct'",
            str(author._extraction_error("length")),
            "the coder replied in Latin",
        ],
    )

    assert breakdown.total == 4
    assert breakdown.counts["import_error"] == 1
    assert breakdown.counts["api_misuse"] == 1
    assert breakdown.counts["truncated"] == 1
    assert breakdown.unclassified_share == 0.25


def test_the_pure_classifier_and_a_registered_taxonomy_agree_on_every_class() -> None:
    taxonomy = _registered()
    errors = [
        "ModuleNotFoundError: No module named 'talib'",
        "AtrState.update() missing 1 required positional argument: 'bar'",
        "class name attribute 'Probe' != file name 'probe'",
        "signals() disagrees with the on_bar replay on the fixture (parity violation)",
        "scenario 'chop': non-deterministic replay — targets differ at bar 88",
        "scenario 'chop': never long in [80, 120]",
        str(author._extraction_error("length")),
        str(author._extraction_error("end_turn")),
        f"declared warmup_bars=400 is {WARMUP_TOO_LARGE_MARKER}: no room",
        "who knows",
    ]

    assert [taxonomy.classify_one("coder", error) for error in errors] == [
        classify_coder_failure(error) for error in errors
    ]


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


def test_the_coder_taxonomy_module_reaches_no_io_no_clock_and_no_randomness() -> None:
    """Classification is a pure function of the string it is handed — the mechanism's own rule."""
    assert _imports(MODULE_SOURCE) <= {"__future__", "dataclasses", "re", "typing", "noctis"}
    text = MODULE_SOURCE.read_text(encoding="utf-8")
    for forbidden in ("datetime", "utcnow", "open(", "Path(", "os.", "random.", "time.", "json."):
        assert forbidden not in text, forbidden


def _module_level_bindings(source: Path) -> dict[str, ast.expr]:
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


def test_the_coder_taxonomy_module_binds_no_module_level_mutable_container() -> None:
    containers = sorted(
        name
        for name, value in _module_level_bindings(MODULE_SOURCE).items()
        if isinstance(value, ast.Dict | ast.List | ast.Set | ast.DictComp | ast.ListComp)
    )

    assert not containers, f"module-level mutable state: {containers}"


def test_the_module_level_binding_walk_sees_the_modules_own_vocabulary() -> None:
    """Non-vacuity: the check above only means something if the walk reaches real bindings."""
    assert {"CODER_CLASSES", "CODER_SITE_ID"} <= set(_module_level_bindings(MODULE_SOURCE))
