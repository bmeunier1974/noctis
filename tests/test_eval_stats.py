"""Paired statistics (#200): the tests are the hand-worked tables, and the refusals.

Three things are asserted here, in the order they matter.

**The arithmetic is checkable by hand.** McNemar's exact conditional binomial is verified against
contingency tables whose p-values are written out as fractions in the test body, and the paired
permutation test is verified against sign-vector enumerations small enough to count on paper (four
identical deltas: two of sixteen sign vectors are as extreme, so p = 0.125). A statistic nobody can
re-derive from the test file is a statistic nobody should trust a promotion to.

**The refusals are the feature.** A comparison the module cannot qualify — mismatched case sets, a
missing per-case outcome, no discordant pairs, an empty case set, a pass count above the rep count —
returns a :class:`~noctis.eval.stats.Refusal` naming the reason instead of a number. The tests hold
that posture at the boundary: never a silent intersection of two case sets, never a dropped case.

**The verdict is tri-state, and the module is pure.** Following the parity module's flip assessment
(``src/noctis/research/parity.py``): an ``n/a`` input makes the affected half inconclusive rather
than deciding it, ratios are denominator-guarded to ``None``, and ``None`` renders as the literal
``n/a`` — spelled not here but once, in :mod:`noctis.eval.reading`, the eval layer's shared reading
vocabulary this module formats its cells through (#303). That is asserted structurally, beside the
import allowlist (stdlib plus that one pure module — no engine, no I/O, no clock) and the
seeded-randomness inspection that every use of ``random`` goes through an explicitly seeded
generator.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Sequence
from pathlib import Path

import pytest

from noctis.eval import stats
from noctis.eval.stats import (
    FAIL,
    INCONCLUSIVE,
    MCNEMAR,
    PAIRED_PERMUTATION,
    PASS,
    Comparison,
    Refusal,
    assess,
    bootstrap_ci,
    compare,
    mcnemar_exact_p,
    measure_stability_band,
    paired_permutation,
    render_comparison,
    render_result,
)

STATS_SOURCE = Path(str(stats.__file__))


def _cases(outcomes: Sequence[int | None]) -> dict[str, int | None]:
    """Per-case outcomes keyed by a zero-padded case id, so sorted order is authoring order."""
    return {f"case-{index:02d}": outcome for index, outcome in enumerate(outcomes)}


def _discordant_pair(favouring_a: int, favouring_b: int, concordant: int = 0) -> tuple[dict, dict]:
    """A single-rep pair whose contingency table is exactly ``(b, c)`` plus both-pass cases."""
    a = [1] * favouring_a + [0] * favouring_b + [1] * concordant
    b = [0] * favouring_a + [1] * favouring_b + [1] * concordant
    return _cases(a), _cases(b)


def _qualified(reps: int = 1, seed: int = 7) -> Comparison:
    """A comparison that qualifies: ten cases favouring B, two both-pass, single rep."""
    a, b = _discordant_pair(0, 10, concordant=2)
    result = compare(a, b, reps=reps, seed=seed)
    assert isinstance(result, Comparison)
    return result


# ── McNemar, against hand-worked contingency tables ───────────────────────────────────────


def test_mcnemar_matches_the_hand_worked_table_with_eight_and_two_discordant_pairs() -> None:
    # b=8, c=2 -> 10 discordant pairs. 2 * P(X <= 2 | X ~ Binomial(10, 1/2))
    #           = 2 * (1 + 10 + 45) / 1024 = 112 / 1024.
    assert mcnemar_exact_p(8, 2) == pytest.approx(112 / 1024)


def test_mcnemar_matches_the_hand_worked_table_with_a_one_sided_five_zero_split() -> None:
    # b=0, c=5 -> 2 * P(X <= 0 | X ~ Binomial(5, 1/2)) = 2 * 1/32 = 0.0625.
    assert mcnemar_exact_p(0, 5) == pytest.approx(0.0625)


def test_mcnemar_caps_a_perfectly_balanced_discordant_split_at_one() -> None:
    # b=c=3 -> 2 * P(X <= 3 | X ~ Binomial(6, 1/2)) = 2 * 42/64 = 1.3125, capped at 1.0.
    assert mcnemar_exact_p(3, 3) == 1.0


def test_mcnemar_guards_a_zero_discordant_denominator_to_none() -> None:
    assert mcnemar_exact_p(0, 0) is None


def test_mcnemar_refuses_a_negative_discordant_count() -> None:
    with pytest.raises(ValueError, match="discordant"):
        mcnemar_exact_p(-1, 4)


def test_a_single_rep_comparison_reports_the_mcnemar_p_value_of_its_contingency_table() -> None:
    a, b = _discordant_pair(8, 2, concordant=10)

    result = compare(a, b, reps=1, seed=3)

    assert isinstance(result, Comparison)
    assert result.discordant == 10
    assert result.p_value == pytest.approx(112 / 1024)


# ── test selection is explicit ────────────────────────────────────────────────────────────


def test_a_single_rep_comparison_selects_mcnemar() -> None:
    a, b = _discordant_pair(2, 6)

    result = compare(a, b, reps=1, seed=3)

    assert isinstance(result, Comparison)
    assert result.test == MCNEMAR
    assert result.permutations is None


def test_a_multi_rep_comparison_selects_the_paired_permutation_test_as_primary() -> None:
    a = _cases([0, 1, 1, 0, 2, 1])
    b = _cases([3, 2, 1, 2, 2, 3])

    result = compare(a, b, reps=3, seed=3)

    assert isinstance(result, Comparison)
    assert result.test == PAIRED_PERMUTATION
    assert result.permutations == 2**6


def test_every_comparison_names_its_case_count_its_reps_and_the_test_it_used() -> None:
    a = _cases([1, 0, 1, 0, 0, 1])
    b = _cases([0, 1, 1, 1, 0, 0])

    for reps in (1, 4):
        result = compare(a, b, reps=reps, seed=5)

        assert isinstance(result, Comparison)
        assert result.n == 6
        assert result.reps == reps
        assert result.test == (MCNEMAR if reps == 1 else PAIRED_PERMUTATION)


# ── the paired permutation test ───────────────────────────────────────────────────────────


def test_the_permutation_test_enumerates_every_sign_vector_for_a_small_case_count() -> None:
    # Four identical deltas: of the 16 sign vectors only all-plus and all-minus reach |mean| = 1.
    assert paired_permutation([1.0, 1.0, 1.0, 1.0], seed=1) == pytest.approx(2 / 16)


def test_the_permutation_test_enumerates_a_three_case_mixture_by_hand() -> None:
    # (+1, +1, -1) has mean 1/3, and every one of the 8 sign vectors reaches |mean| >= 1/3.
    assert paired_permutation([1.0, 1.0, -1.0], seed=1) == pytest.approx(1.0)


def test_the_permutation_test_returns_one_when_a_single_case_makes_every_flip_as_extreme() -> None:
    assert paired_permutation([0.5], seed=1) == pytest.approx(1.0)


def test_the_permutation_test_guards_an_all_zero_delta_vector_to_none() -> None:
    assert paired_permutation([0.0, 0.0, 0.0], seed=1) is None


def test_the_permutation_test_guards_an_empty_delta_vector_to_none() -> None:
    assert paired_permutation([], seed=1) is None


_SAMPLED_DELTAS = [
    1.0, 1.0, -1.0, 0.5, 0.5, -0.5, 1.0, 0.0,
    0.5, -1.0, 1.0, 0.5, -0.5, 1.0, 0.0, 0.5,
]  # fmt: skip
"""Sixteen cases: 2**16 sign vectors exceed the default budget, so the test samples."""


def test_the_permutation_test_is_deterministic_under_a_fixed_seed() -> None:
    first = paired_permutation(_SAMPLED_DELTAS, seed=11)
    second = paired_permutation(_SAMPLED_DELTAS, seed=11)

    assert first == second
    assert first is not None and 0.0 < first < 1.0


def test_the_permutation_test_varies_across_seeds_once_it_samples() -> None:
    sampled = {paired_permutation(_SAMPLED_DELTAS, seed=seed) for seed in range(6)}

    assert len(sampled) > 1


# ── the bootstrap interval over cases ─────────────────────────────────────────────────────


def test_the_bootstrap_interval_collapses_onto_the_only_case_when_there_is_one() -> None:
    assert bootstrap_ci([0.25], seed=3) == (0.25, 0.25)


def test_the_bootstrap_interval_is_zero_wide_when_every_case_agrees() -> None:
    assert bootstrap_ci([0.0] * 8, seed=3) == (0.0, 0.0)


def test_the_bootstrap_interval_is_one_wide_when_every_case_flips() -> None:
    assert bootstrap_ci([1.0] * 8, seed=3) == (1.0, 1.0)


def test_the_bootstrap_interval_guards_an_empty_case_set_to_none() -> None:
    assert bootstrap_ci([], seed=3) is None


def test_the_bootstrap_interval_is_deterministic_under_a_fixed_seed() -> None:
    deltas = [1.0, 0.0, 1.0, 0.5, -0.5, 1.0, 0.0, 0.5]

    assert bootstrap_ci(deltas, seed=9) == bootstrap_ci(deltas, seed=9)


def test_the_bootstrap_interval_brackets_the_observed_delta() -> None:
    deltas = [1.0] * 6 + [0.0] * 4  # observed delta 0.6

    interval = bootstrap_ci(deltas, seed=5)

    assert interval is not None
    low, high = interval
    assert low < 0.6 < high
    assert 0.0 <= low and high <= 1.0


def test_the_comparison_carries_the_bootstrap_interval_of_its_delta() -> None:
    result = _qualified()

    assert result.ci_low is not None and result.ci_high is not None
    assert result.ci_low <= result.delta <= result.ci_high


# ── the run-twice stability band ──────────────────────────────────────────────────────────


def test_the_stability_band_of_two_identical_runs_is_zero() -> None:
    run = _cases([1, 0, 1, 1, 0])

    band = measure_stability_band(run, dict(run))

    assert not isinstance(band, Refusal)
    assert band.band == 0.0
    assert (band.n, band.reps) == (5, 1)


def test_the_stability_band_is_the_absolute_pass_rate_gap_between_two_same_config_runs() -> None:
    first = _cases([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])  # 0.4
    second = _cases([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])  # 0.6

    band = measure_stability_band(first, second)

    assert not isinstance(band, Refusal)
    assert band.band == pytest.approx(0.2)
    assert (band.rate_first, band.rate_second) == (pytest.approx(0.4), pytest.approx(0.6))


def test_the_stability_band_of_two_all_pass_runs_is_zero() -> None:
    band = measure_stability_band(_cases([3, 3, 3]), _cases([3, 3, 3]), reps=3)

    assert not isinstance(band, Refusal)
    assert band.band == 0.0


def test_the_stability_band_of_two_all_fail_runs_is_zero() -> None:
    band = measure_stability_band(_cases([0, 0, 0]), _cases([0, 0, 0]), reps=3)

    assert not isinstance(band, Refusal)
    assert band.band == 0.0


def test_the_stability_band_refuses_two_runs_over_different_case_sets() -> None:
    band = measure_stability_band(_cases([1, 0]), {"case-00": 1, "case-09": 0})

    assert isinstance(band, Refusal)
    assert band.reason == stats.MISMATCHED_CASES


def test_the_stability_band_refuses_an_empty_pair_of_runs() -> None:
    band = measure_stability_band({}, {})

    assert isinstance(band, Refusal)
    assert band.reason == stats.NO_CASES


# ── refusals: what the module will not qualify ────────────────────────────────────────────


def test_a_comparison_over_mismatched_case_sets_refuses_and_names_the_offending_cases() -> None:
    a = {"case-00": 1, "case-01": 0}
    b = {"case-00": 0, "case-02": 1}

    result = compare(a, b, reps=1, seed=1)

    assert isinstance(result, Refusal)
    assert result.reason == stats.MISMATCHED_CASES
    assert "case-01" in result.detail and "case-02" in result.detail


def test_a_comparison_with_a_missing_per_case_outcome_refuses_rather_than_dropping_it() -> None:
    a = _cases([1, 0, None, 1])
    b = _cases([0, 1, 1, 0])

    result = compare(a, b, reps=1, seed=1)

    assert isinstance(result, Refusal)
    assert result.reason == stats.MISSING_OUTCOMES
    assert "case-02" in result.detail


def test_a_comparison_with_no_discordant_cases_refuses_for_want_of_discordant_pairs() -> None:
    run = _cases([1, 1, 0, 1, 0])

    result = compare(run, dict(run), reps=1, seed=1)

    assert isinstance(result, Refusal)
    assert result.reason == stats.NO_DISCORDANT_PAIRS


def test_a_comparison_over_an_empty_case_set_refuses() -> None:
    result = compare({}, {}, reps=1, seed=1)

    assert isinstance(result, Refusal)
    assert result.reason == stats.NO_CASES


def test_a_comparison_whose_pass_count_exceeds_its_rep_count_refuses() -> None:
    result = compare(_cases([1, 4]), _cases([0, 1]), reps=3, seed=1)

    assert isinstance(result, Refusal)
    assert result.reason == stats.OUT_OF_RANGE
    assert "case-01" in result.detail


def test_a_comparison_with_fewer_than_one_rep_refuses() -> None:
    result = compare(_cases([1, 0]), _cases([0, 1]), reps=0, seed=1)

    assert isinstance(result, Refusal)
    assert result.reason == stats.INVALID_REPS


def test_rendering_a_refusal_names_the_reason_and_emits_no_statistic() -> None:
    refused = compare({}, {}, reps=1, seed=1)

    text = render_result(refused)

    assert text.startswith("REFUSED")
    assert stats.NO_CASES in text
    assert "p-value" not in text
    assert "Delta" not in text


# ── determinism of the comparison itself ──────────────────────────────────────────────────


def test_a_comparison_is_unchanged_when_the_case_mappings_are_ordered_differently() -> None:
    a = _cases([0, 1, 1, 0, 2, 1])
    b = _cases([3, 2, 1, 2, 2, 3])
    reversed_a = dict(reversed(list(a.items())))

    assert compare(reversed_a, b, reps=3, seed=4) == compare(a, b, reps=3, seed=4)


# ── the tri-state verdict, following the parity prior art ─────────────────────────────────


def test_a_significant_improvement_outside_the_stability_band_passes() -> None:
    result = _qualified()

    assessment = assess(result, band=0.05)

    assert (assessment.improved, assessment.significant, assessment.resolvable) == (
        True,
        True,
        True,
    )
    assert assessment.verdict == PASS
    assert assessment.summary.startswith("PASS")


def test_an_improvement_the_test_cannot_call_significant_fails() -> None:
    a, b = _discordant_pair(2, 3, concordant=5)  # p = 1.0 on five discordant pairs

    result = compare(a, b, reps=1, seed=6)
    assert isinstance(result, Comparison)
    assessment = assess(result, band=0.01)

    assert assessment.improved is True
    assert assessment.significant is False
    assert assessment.verdict == FAIL
    assert assessment.summary.startswith("FAIL")


def test_a_regression_fails_even_when_the_test_calls_it_significant() -> None:
    a, b = _discordant_pair(10, 0, concordant=2)

    result = compare(a, b, reps=1, seed=6)
    assert isinstance(result, Comparison)
    assessment = assess(result, band=0.05)

    assert assessment.improved is False
    assert assessment.significant is True
    assert assessment.verdict == FAIL


def test_a_delta_inside_the_stability_band_is_declared_unresolvable() -> None:
    result = _qualified()

    assessment = assess(result, band=0.95)

    assert assessment.resolvable is False
    assert assessment.verdict == INCONCLUSIVE
    assert "stability band" in assessment.summary


def test_an_unmeasured_stability_band_leaves_the_resolvability_half_inconclusive() -> None:
    result = _qualified()

    assessment = assess(result)

    assert assessment.resolvable is None
    assert assessment.verdict == INCONCLUSIVE
    assert "stability band" in assessment.summary


def test_an_n_a_p_value_leaves_the_significance_half_inconclusive() -> None:
    comparison = Comparison(
        n=10,
        reps=1,
        test=MCNEMAR,
        rate_a=0.4,
        rate_b=0.6,
        delta=0.2,
        p_value=None,
        ci_low=0.05,
        ci_high=0.35,
        confidence=0.95,
        discordant=4,
        permutations=None,
    )

    assessment = assess(comparison, band=0.01)

    assert assessment.significant is None
    assert assessment.verdict == INCONCLUSIVE
    assert "p-value" in assessment.summary


# ── rendering ─────────────────────────────────────────────────────────────────────────────


def test_the_rendered_comparison_names_the_case_count_reps_and_the_test() -> None:
    result = _qualified()

    text = render_comparison(result, assess(result, band=0.05))

    assert "n=12" in text
    assert "reps=1" in text
    assert f"test={MCNEMAR}" in text


def test_the_rendered_comparison_carries_the_verdict_summary() -> None:
    result = _qualified()
    assessment = assess(result, band=0.05)

    assert assessment.summary in render_comparison(result, assessment)


def test_an_unmeasured_stability_band_renders_as_the_literal_n_a() -> None:
    result = _qualified()

    text = render_comparison(result, assess(result))

    assert "n/a" in text


def test_rendering_a_qualified_result_goes_through_the_comparison_table() -> None:
    result = _qualified()

    assert render_result(result, band=0.05) == render_comparison(result, assess(result, band=0.05))


# ── purity and seeded randomness, structurally ────────────────────────────────────────────


def _imports(source: Path) -> set[str]:
    tree = ast.parse(source.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


def test_the_stats_module_reaches_no_io_no_clock_and_no_engine() -> None:
    """The load-bearing decision of this slice: the statistics are a pure function of their
    inputs. Nothing here reads a file, a clock, the settings — or the engine.

    The one ``noctis`` module it reaches for is :mod:`noctis.eval.reading`, the eval layer's own
    stdlib-only rendering vocabulary, which computes no figure and is held to that purity by
    ``tests/test_eval_reading.py``."""
    assert _imports(STATS_SOURCE) <= {
        "__future__",
        "collections",
        "dataclasses",
        "itertools",
        "math",
        "noctis",
        "random",
        "typing",
    }
    tree = ast.parse(STATS_SOURCE.read_text())
    reached = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("noctis")
    }
    assert reached == {"noctis.eval.reading"}

    text = STATS_SOURCE.read_text()
    for forbidden in ("open(", "Path(", "os.", "json.", "datetime", "time.time"):
        assert forbidden not in text, forbidden


def test_every_use_of_the_random_module_goes_through_an_explicitly_seeded_generator() -> None:
    """``random.random()`` and friends read a process-global generator nobody seeded. Every
    attribute this module takes off ``random`` must therefore be ``Random`` itself."""
    tree = ast.parse(STATS_SOURCE.read_text())
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "random"
    }

    assert used == {"Random"}


def test_the_entry_points_that_use_randomness_require_a_seed() -> None:
    for entry_point in (compare, paired_permutation, bootstrap_ci):
        seed = inspect.signature(entry_point).parameters["seed"]

        assert seed.default is inspect.Parameter.empty, entry_point.__name__
        assert seed.kind is inspect.Parameter.KEYWORD_ONLY, entry_point.__name__


def test_none_is_not_spelled_here_at_all_because_the_reading_vocabulary_owns_the_word() -> None:
    """The stronger form of "in exactly one place": the one place is no longer this module.

    Every cell goes through :func:`noctis.eval.reading.fmt`, so this module cannot disagree with the
    coder's reading or the bench report about what a missing figure looks like — and the rendering
    of one is pinned as behaviour above."""
    tree = ast.parse(STATS_SOURCE.read_text())
    literals = [
        node for node in ast.walk(tree) if isinstance(node, ast.Constant) and node.value == "n/a"
    ]

    assert literals == []
