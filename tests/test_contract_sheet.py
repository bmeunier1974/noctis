"""The API contract sheet + its drift guard.

The sheet is a hand-written data table rendered to a deterministic prompt block (no runtime
introspection), so these tests read the rendered text like a coder would. The drift guard is
the load-bearing one: it walks every name/signature the table declares and asserts it against
the live scenario/indicator/exit modules via ``inspect.signature``, so the sheet can never
silently rot away from the code the write gate grades against.
"""

from __future__ import annotations

import importlib
import inspect

from noctis.research import contract_sheet as cs
from noctis.strategies import scenarios


# ── the rendered sheet is deterministic and covers the whole surface ──────────────────────
def test_render_is_deterministic_and_matches_the_constant():
    assert cs.render_contract_sheet() == cs.render_contract_sheet()
    assert cs.CONTRACT_SHEET == cs.render_contract_sheet()


def test_sheet_renders_every_declared_signature():
    sheet = cs.CONTRACT_SHEET
    for section in cs.SECTIONS:
        for entry in section.entries:
            assert entry.signature() in sheet, f"{entry.name} signature missing from the sheet"


def test_sheet_names_the_scenario_builder_closure():
    # The gate has exactly seven builders; the sheet must foreclose invention of others.
    sheet = cs.CONTRACT_SHEET
    for builder in ("flat(", "trend(", "selloff(", "recovery(", "chop(", "vol_spike(", "gap("):
        assert builder in sheet
    assert "gap" in sheet and "no bars" in sheet.lower()  # gap adds nothing to tape length
    # An explicit closure statement so the coder does not reach for a builder that isn't there.
    assert "only" in sheet.lower()


def test_sheet_covers_every_expectation_including_the_zero_arg_one():
    sheet = cs.CONTRACT_SHEET
    for exp in (
        "flat_until(",
        "long_within(",
        "holds_long_through(",
        "short_within(",
        "holds_short_through(",
        "flat_by(",
    ):
        assert exp in sheet
    assert "always_flat()" in sheet  # rendered with its zero-arg call


def test_sheet_states_the_tape_shape_rules():
    sheet = cs.CONTRACT_SHEET
    assert f"{scenarios.MIN_SCENARIOS}-{scenarios.MAX_SCENARIOS}" in sheet
    assert f"{scenarios.MIN_SCENARIO_BARS}-{scenarios.MAX_SCENARIO_BARS}" in sheet
    assert "always_flat()" in sheet
    assert "directional" in sheet.lower()


def test_sheet_states_warmup_semantics_and_the_update_convention():
    sheet = cs.CONTRACT_SHEET
    low = sheet.lower()
    assert "none" in low and "warmup" in low  # tail funcs return None during warmup
    assert "nan" in low  # State classes return nan during warmup
    assert "guard" in low  # always guard the warmup return
    assert ".update(bar)" in sheet  # the calling convention
    # The documented float-updating exception is called out, not silently identical.
    assert "ZScoreState(" in sheet
    assert ".update(x)" in sheet or "update(x" in sheet


def test_sheet_states_exit_fields_as_fractions_of_entry():
    sheet = cs.CONTRACT_SHEET
    assert "ExitRules(stop_pct=None, take_profit_pct=None, trail_pct=None)" in sheet
    assert "fraction" in sheet.lower() and "entry" in sheet.lower()


# ── the drift guard: every declared name/signature must match the live modules ────────────
def _params_without_self(sig: inspect.Signature) -> list[inspect.Parameter]:
    return [p for p in sig.parameters.values() if p.name != "self"]


def test_every_declared_signature_matches_the_live_module():
    for section in cs.SECTIONS:
        module = importlib.import_module(section.module_name)
        for entry in section.entries:
            obj = getattr(module, entry.name)  # AttributeError = the sheet names a ghost
            live = _params_without_self(inspect.signature(obj))
            assert [p.name for p in live] == [p.name for p in entry.params], (
                f"{section.module_name}.{entry.name}: parameter names drifted"
            )
            for lp, ep in zip(live, entry.params, strict=True):
                if ep.default is cs.REQUIRED:
                    assert lp.default is inspect.Parameter.empty, (
                        f"{entry.name}.{ep.name} gained a default upstream"
                    )
                else:
                    assert lp.default == ep.default, (
                        f"{entry.name}.{ep.name} default drifted: {lp.default!r} != {ep.default!r}"
                    )


def test_update_calling_convention_matches_the_live_state_classes():
    module = importlib.import_module("noctis.strategies.indicators")
    for section in cs.SECTIONS:
        for entry in section.entries:
            if entry.update_arg is None:
                continue
            cls = getattr(module, entry.name)
            live = _params_without_self(inspect.signature(cls.update))
            assert live, f"{entry.name}.update takes no argument"
            assert live[0].name == entry.update_arg, (
                f"{entry.name}.update first arg drifted: {live[0].name!r} != {entry.update_arg!r}"
            )


def test_tape_constants_match_the_live_scenario_module():
    for const in cs.TAPE_CONSTANTS:
        assert getattr(scenarios, const.live_name) == const.value, (
            f"tape constant {const.live_name} drifted from the sheet"
        )


# ── retry-error enrichment: hints render from the same rows the sheet renders ─────────────
def _entry(name: str) -> cs.Entry:
    return next(e for section in cs.SECTIONS for e in section.entries if e.name == name)


def test_state_update_arity_error_maps_to_the_state_row_signature():
    # A State .update() arity mistake resolves the class name against the State-class rows and
    # renders a hint from that row — its constructor signature and float/Bar update convention.
    hint = cs.hint_for_gate_error("AtrState.update() takes 2 positional arguments but 4 were given")
    assert hint is not None
    entry = _entry("AtrState")
    assert entry.signature() in hint  # AtrState(period)
    assert f".update({entry.update_arg})" in hint  # .update(bar)


def test_state_update_missing_argument_error_also_maps_to_the_state_row():
    # The other arity shape (too few args) is the same class of mistake and the same hint.
    hint = cs.hint_for_gate_error("AtrState.update() missing 1 required positional argument: 'bar'")
    assert hint is not None
    assert _entry("AtrState").signature() in hint


def test_unknown_exit_rules_field_maps_to_the_exit_row_signature():
    hint = cs.hint_for_gate_error(
        "ExitRules.__init__() got an unexpected keyword argument 'target_pct'"
    )
    assert hint is not None
    assert _entry("ExitRules").signature() in hint


def test_scenario_builder_bad_kwarg_maps_to_the_builder_row_signature():
    # The builder error arrives wrapped (scenarios() raised TypeError: trend() got ...); the
    # enricher still resolves the builder name and renders that row's signature.
    hint = cs.hint_for_gate_error(
        "scenarios() raised TypeError: trend() got an unexpected keyword argument 'drift'"
    )
    assert hint is not None
    assert _entry("trend").signature() in hint  # trend(n, pct)


def test_unexpected_kwarg_resolves_any_table_row_not_just_exit_rules():
    # The keyword-mismatch pattern is general: an expectation/tail-function name resolves too.
    hint = cs.hint_for_gate_error(
        "scenarios() raised TypeError: long_within() got an unexpected keyword argument 'bad'"
    )
    assert hint is not None
    assert _entry("long_within").signature() in hint


def test_unknown_pattern_yields_no_hint():
    # Errors matching no known helper-API pattern get no hint — the retry keeps its raw behavior.
    assert (
        cs.hint_for_gate_error("class sets name='mismatch' but the strategy/file name is 'probe'")
        is None
    )
    assert cs.hint_for_gate_error("on_bar replay produced a short target series") is None
    assert cs.hint_for_gate_error("") is None


def test_matched_shape_for_a_ghost_name_yields_no_hint():
    # A well-formed pattern naming a callable that is NOT in the table stays unenriched (the
    # single source of truth cannot invent a signature it does not hold).
    assert (
        cs.hint_for_gate_error("not_a_real_helper() got an unexpected keyword argument 'x'") is None
    )
    assert (
        cs.hint_for_gate_error("GhostState.update() takes 2 positional arguments but 4 were given")
        is None
    )
