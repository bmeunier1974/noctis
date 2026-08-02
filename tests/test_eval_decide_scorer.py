"""The DECIDE scorer family (#206): approval-side agreement paired with approval rate.

DECIDE is the judgment that turns a tuned candidate into promote/reject, and v1 measures it on the
**approval side**: of the candidates the model approved, what share did the promotion gates then
promote. Three properties carry this suite:

* **The pair is structural.** Agreement and approval rate are one co-primary value, and there is no
  public path to the first without the second — a config that approves one candidate a year and is
  right about it scores a perfect agreement, and must be visibly throughput-poor in the same breath.
* **A revise is a deferral, not a judgment.** It is excluded from agreement's numerator and
  denominator and reported as its own rate, beside the share of revises whose eventual verdict
  differed from the proposal that earned the re-ask.
* **Absences are ``n/a``, never zero.** An approval the gates never labelled lands in an explicit
  excluded count and moves agreement neither way; an empty batch yields ``None``-carrying values.

Every figure is asserted against a hand-computed fixture, and the module is held to the pure-module
inspection the bench metrics are held to.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from noctis.eval import decide_scorer, stats
from noctis.eval.decide_scorer import (
    APPROVE,
    REJECT,
    REVISE,
    VERDICTS,
    DecideMetrics,
    DecideOutcome,
    GateLabel,
    promotion_outcomes,
    score_decide_batch,
)

SCORER_SOURCE = Path(__file__).resolve().parents[1] / "src" / "noctis" / "eval" / "decide_scorer.py"

PROMOTED, REFUSED, UNLABELED = GateLabel.PROMOTED, GateLabel.REFUSED, GateLabel.UNLABELED


def _decided(
    case_id: str,
    verdict: str,
    label: GateLabel = UNLABELED,
    *,
    revised: bool = False,
    flipped: bool = False,
) -> DecideOutcome:
    """One scored DECIDE outcome, spelled the way a batch fixture reads."""
    return DecideOutcome(
        case_id=case_id,
        verdict=verdict,
        label=label,
        revised=revised,
        revise_flipped=flipped,
    )


# The headline fixture, hand-computable end to end: ten decided candidates, six approved, five of
# those approvals labelled by the gates and four of the five promoted, three revises of which one
# flipped the verdict it started from.
MIXED = (
    _decided("momentum_a", APPROVE, PROMOTED),
    _decided("momentum_b", APPROVE, PROMOTED),
    _decided("meanrev_a", APPROVE, PROMOTED),
    _decided("meanrev_b", APPROVE, REFUSED),
    _decided("breakout_a", APPROVE),
    _decided("breakout_b", REJECT),
    _decided("carry_a", REJECT, REFUSED),
    _decided("carry_b", REVISE, revised=True),
    _decided("vol_a", REJECT, REFUSED, revised=True, flipped=True),
    _decided("vol_b", APPROVE, PROMOTED, revised=True),
)

# Approvals: 6 of 10. Labelled approvals: 5, of which 4 promoted. Revises: 3, of which 1 flipped.


# ── the four rates, against hand-computed fixtures ────────────────────────────────────────
def test_approval_side_agreement_is_the_promoted_share_of_the_labeled_approvals():
    """Four of the five approvals the gates labelled were promoted — the unlabelled sixth is not
    in the denominator, and the rejections are not in it either."""
    assert score_decide_batch(MIXED).approval.agreement == pytest.approx(4 / 5)


def test_the_approval_rate_is_the_approved_share_of_every_decided_candidate():
    """Six approvals out of ten decided candidates — the throughput half of the pair."""
    assert score_decide_batch(MIXED).approval.approval_rate == pytest.approx(0.6)


def test_the_revise_rate_is_the_revised_share_of_every_decided_candidate():
    """Three of the ten candidates earned a revise somewhere on the way to their verdict."""
    assert score_decide_batch(MIXED).revise_rate == pytest.approx(0.3)


def test_the_revise_flip_rate_is_the_share_of_revises_whose_eventual_verdict_changed():
    """One of the three revises ended on a different verdict than the one it deferred."""
    assert score_decide_batch(MIXED).revise_flip_rate == pytest.approx(1 / 3)


def test_the_counts_behind_the_rates_are_carried_beside_them():
    """Every denominator a reader would have to reconstruct is on the record instead."""
    metrics = score_decide_batch(MIXED)

    assert (metrics.decided, metrics.rejections, metrics.revises, metrics.revise_flips) == (
        10,
        3,
        3,
        1,
    )
    assert (metrics.approval.approvals, metrics.approval.labeled_approvals) == (6, 5)
    assert (metrics.approval.promoted, metrics.approval.unlabeled_approvals) == (4, 1)


def test_a_batch_that_approved_nothing_reports_a_measured_zero_approval_rate_and_no_agreement():
    """A measured zero (nobody was approved) is a finding; agreement over no approvals is n/a."""
    metrics = score_decide_batch(
        (
            _decided("a", REJECT),
            _decided("b", REJECT, REFUSED),
            _decided("c", REJECT),
        )
    )

    assert metrics.approval.approval_rate == 0.0
    assert metrics.approval.agreement is None


def test_a_batch_of_nothing_but_revises_is_all_deferral_and_carries_no_approval_agreement():
    """Every candidate deferred: the revise rate is 1.0 and there is no judgment to grade."""
    metrics = score_decide_batch(
        (
            _decided("a", REVISE, revised=True),
            _decided("b", REVISE, revised=True),
        )
    )

    assert metrics.revise_rate == 1.0
    assert metrics.approval.approval_rate == 0.0
    assert metrics.approval.agreement is None
    assert metrics.revise_flip_rate == 0.0


def test_a_single_promoted_approval_scores_full_agreement_at_full_throughput():
    """The smallest non-degenerate batch: one approval, promoted."""
    metrics = score_decide_batch((_decided("solo", APPROVE, PROMOTED),))

    assert metrics.approval.agreement == 1.0
    assert metrics.approval.approval_rate == 1.0


def test_a_single_refused_approval_scores_a_measured_zero_agreement():
    """One approval the gates refused: zero agreement is measured, not missing."""
    assert score_decide_batch((_decided("solo", APPROVE, REFUSED),)).approval.agreement == 0.0


def test_an_empty_batch_yields_none_carrying_values_rather_than_zeros():
    """Nothing was decided, so every rate is ``n/a`` — and nothing divides."""
    metrics = score_decide_batch(())

    assert metrics.decided == 0
    assert metrics.approval.agreement is None
    assert metrics.approval.approval_rate is None
    assert metrics.revise_rate is None
    assert metrics.revise_flip_rate is None


# ── a revise is a deferral, not a gradeable judgment ──────────────────────────────────────
def test_a_revise_never_enters_the_agreement_denominator_it_only_raises_the_revise_rate():
    """Adding a deferral to a batch cannot move the agreement of the judgments it made."""
    judged = (_decided("a", APPROVE, PROMOTED), _decided("b", APPROVE, REFUSED))
    deferred = (*judged, _decided("c", REVISE, revised=True))

    assert (
        score_decide_batch(deferred).approval.agreement
        == score_decide_batch(judged).approval.agreement
    )
    assert score_decide_batch(deferred).approval.labeled_approvals == 2
    assert score_decide_batch(deferred).revise_rate == pytest.approx(1 / 3)


def test_a_revise_that_ended_on_the_verdict_it_started_from_is_not_counted_as_a_flip():
    """The re-ask happened and the model held its ground — a revise, and a measured zero flip."""
    metrics = score_decide_batch((_decided("a", APPROVE, PROMOTED, revised=True),))

    assert metrics.revise_rate == 1.0
    assert metrics.revise_flip_rate == 0.0


def test_a_batch_with_no_revises_at_all_reports_a_measured_zero_rate_and_no_flip_rate():
    """Zero revises is a measurement; the flip rate over them has no denominator and is ``n/a``."""
    metrics = score_decide_batch((_decided("a", APPROVE, PROMOTED), _decided("b", REJECT)))

    assert metrics.revise_rate == 0.0
    assert metrics.revise_flip_rate is None


# ── n/a honesty: an unlabelled approval is excluded, not counted either way ───────────────
def test_an_unlabeled_approval_lands_in_the_n_a_count_and_leaves_agreement_untouched():
    """The gates never judged it, so it is neither an agreement nor a disagreement."""
    labeled = (
        _decided("a", APPROVE, PROMOTED),
        _decided("b", APPROVE, REFUSED),
        _decided("c", REJECT),
    )
    with_unlabeled = (*labeled, _decided("d", APPROVE))

    assert score_decide_batch(with_unlabeled).approval.agreement == pytest.approx(0.5)
    assert score_decide_batch(with_unlabeled).approval.unlabeled_approvals == 1
    assert score_decide_batch(with_unlabeled).approval.labeled_approvals == 2


def test_an_approval_the_gates_never_labeled_still_counts_toward_the_approval_rate():
    """Throughput is what the model did; a label is what the gates said about it afterwards."""
    metrics = score_decide_batch((_decided("a", APPROVE), _decided("b", REJECT)))

    assert metrics.approval.approval_rate == 0.5
    assert metrics.approval.agreement is None


def test_a_batch_whose_every_approval_is_unlabeled_reports_no_agreement_rather_than_zero():
    """An entirely unlabelled batch has an ``n/a`` agreement and says how many it excluded."""
    metrics = score_decide_batch((_decided("a", APPROVE), _decided("b", APPROVE)))

    assert metrics.approval.agreement is None
    assert metrics.approval.unlabeled_approvals == 2


def test_a_labeled_rejection_does_not_enter_approval_side_agreement():
    """v1 measures the approval side: what the gates said about a candidate the model rejected is
    recorded, and is not evidence about the approvals."""
    without = (_decided("a", APPROVE, PROMOTED),)
    with_rejections = (*without, _decided("b", REJECT, REFUSED), _decided("c", REJECT, PROMOTED))

    assert score_decide_batch(with_rejections).approval.agreement == 1.0
    assert score_decide_batch(with_rejections).approval.labeled_approvals == 1


# ── the inputs refuse what they cannot mean ───────────────────────────────────────────────
def test_a_verdict_outside_the_production_emit_vocabulary_is_refused():
    with pytest.raises(ValueError, match="verdict"):
        _decided("a", "maybe")


def test_a_revise_flip_without_a_revise_is_refused_as_incoherent():
    with pytest.raises(ValueError, match="revise"):
        _decided("a", APPROVE, PROMOTED, flipped=True)


def test_a_verdict_that_ended_on_revise_must_record_the_revise_it_ended_on():
    with pytest.raises(ValueError, match="revise"):
        _decided("a", REVISE)


def test_a_candidate_that_ended_on_a_revise_may_not_carry_a_gate_label():
    """A deferral never reached the gates, so a gate verdict about it is a fabricated label."""
    with pytest.raises(ValueError, match="label"):
        _decided("a", REVISE, PROMOTED, revised=True)


def test_the_verdict_vocabulary_is_the_production_decide_emit_vocabulary():
    """The scorer grades what the driver can actually emit — asserted here rather than imported,
    so the pure module stays pure and drift is caught in the suite instead."""
    from noctis.research.driver import DECIDE_CONTRACT

    assert set(VERDICTS) == set(DECIDE_CONTRACT.schema["properties"]["verdict"]["enum"])


# ── the pairing rule, structurally ────────────────────────────────────────────────────────
def _every_public_rendering(metrics: DecideMetrics) -> list[str]:
    """Every string this module's public surface can produce from one scored batch: the render of
    each frozen value it hands back, plus any exported function that returns text about one."""
    texts = [
        value.render()
        for value in (metrics, metrics.approval)
        if callable(getattr(value, "render", None))
    ]
    for name in decide_scorer.__all__:
        exported = getattr(decide_scorer, name)
        if not inspect.isfunction(exported):
            continue
        for argument in (metrics, metrics.approval, MIXED):
            try:
                produced = exported(argument)
            except Exception:  # a signature that does not take this argument is not a render path
                continue
            if isinstance(produced, str):
                texts.append(produced)
    return texts


def test_agreement_and_the_approval_rate_arrive_as_one_co_primary_pair():
    """One frozen value holds both, so neither can be quoted without the other beside it."""
    pair = score_decide_batch(MIXED).approval

    assert is_dataclass(pair)
    assert {"agreement", "approval_rate"} <= {field.name for field in fields(pair)}


def test_the_metrics_record_carries_no_agreement_of_its_own_beside_the_pair():
    """The only way to the number is through the pair — there is no flattened copy to grab."""
    metrics = score_decide_batch(MIXED)

    assert not hasattr(metrics, "agreement")
    assert "agreement" not in {field.name for field in fields(metrics)}


def test_no_exported_name_offers_the_agreement_number_on_its_own():
    """No standalone ``approval_agreement`` function, and no exported value that carries agreement
    without carrying the approval rate with it."""
    for name in decide_scorer.__all__:
        exported = getattr(decide_scorer, name)
        assert not inspect.isfunction(exported) or "agreement" not in name.lower()
        if is_dataclass(exported):
            carried = {field.name for field in fields(exported)}
            assert "agreement" not in carried or "approval_rate" in carried


def test_every_public_rendering_that_names_agreement_names_the_approval_rate_beside_it():
    """The pairing rule on the render path: there is no way to print one number alone."""
    renderings = _every_public_rendering(score_decide_batch(MIXED))

    assert renderings
    assert any("approval-side agreement" in text.lower() for text in renderings)
    for text in renderings:
        if "approval-side agreement" in text.lower():
            assert "approval rate" in text.lower()


def test_a_config_that_approves_almost_nothing_is_visibly_throughput_poor_beside_its_agreement():
    """The whole point of the pair: perfect agreement over one approval in twenty reads as one
    approval in twenty, in the same block of text."""
    rejected = tuple(_decided(f"r{index}", REJECT) for index in range(19))
    slam_dunk = (_decided("one", APPROVE, PROMOTED), *rejected)
    metrics = score_decide_batch(slam_dunk)

    assert metrics.approval.agreement == 1.0
    assert metrics.approval.approval_rate == pytest.approx(0.05)
    rendered = metrics.render()
    assert "1.0000" in rendered and "0.0500" in rendered


# ── the rendered vocabulary ───────────────────────────────────────────────────────────────
def test_the_rendered_metrics_call_the_number_approval_side_agreement():
    assert "approval-side agreement" in score_decide_batch(MIXED).render().lower()


def test_the_rendered_metrics_report_the_revise_rates_beside_the_pair():
    rendered = score_decide_batch(MIXED).render().lower()

    assert "revise rate" in rendered
    assert "flip" in rendered


def test_an_absent_agreement_renders_as_n_a_rather_than_a_zero():
    """A batch nobody labelled prints ``n/a`` where the number would go."""
    rendered = score_decide_batch((_decided("a", APPROVE),)).render()

    assert "n/a" in rendered


def test_the_excluded_unlabeled_approvals_are_named_in_the_rendering():
    """The count that was kept out of the denominator is on the page, not implied."""
    rendered = score_decide_batch(MIXED).render().lower()

    assert "unlabeled" in rendered


# ── comparing two configurations rides the paired statistics, restating none of it ────────
def test_the_promotion_outcomes_of_a_batch_are_its_approvals_graded_by_the_gates():
    """1 for a promotion, 0 for a refusal — the per-case binary the paired tests pair on."""
    outcomes = promotion_outcomes(MIXED)

    assert outcomes["momentum_a"] == 1
    assert outcomes["meanrev_b"] == 0


def test_a_candidate_the_model_did_not_approve_is_not_part_of_the_approval_side_population():
    """Approval-side agreement is about approvals; a rejection or a deferral is not a case in it."""
    outcomes = promotion_outcomes(MIXED)

    assert "breakout_b" not in outcomes
    assert "carry_b" not in outcomes


def test_an_approval_the_gates_never_labeled_is_present_with_no_outcome_rather_than_dropped():
    """Which cases were compared is part of what a comparison means, so the hole is visible."""
    assert promotion_outcomes(MIXED)["breakout_a"] is None


def test_two_batches_that_approved_the_same_candidates_compare_through_the_paired_statistics():
    """The comparison is :mod:`noctis.eval.stats`' own machinery over the scorer's outcomes."""
    a = (
        _decided("x", APPROVE, PROMOTED),
        _decided("y", APPROVE, REFUSED),
        _decided("z", APPROVE, REFUSED),
    )
    b = (
        _decided("x", APPROVE, PROMOTED),
        _decided("y", APPROVE, PROMOTED),
        _decided("z", APPROVE, PROMOTED),
    )

    result = stats.compare(promotion_outcomes(a), promotion_outcomes(b), seed=7)

    assert isinstance(result, stats.Comparison)
    assert result.rate_a == pytest.approx(1 / 3)
    assert result.rate_b == pytest.approx(1.0)


def test_two_batches_that_approved_different_candidates_refuse_the_paired_comparison():
    """Two configurations that approved different things were not measured on one population."""
    a = (_decided("x", APPROVE, PROMOTED),)
    b = (_decided("y", APPROVE, PROMOTED),)

    result = stats.compare(promotion_outcomes(a), promotion_outcomes(b), seed=7)

    assert isinstance(result, stats.Refusal)
    assert result.reason == stats.MISMATCHED_CASES


def test_an_unlabeled_approval_refuses_the_comparison_rather_than_shrinking_the_population():
    a = (_decided("x", APPROVE, PROMOTED), _decided("y", APPROVE))
    b = (_decided("x", APPROVE, REFUSED), _decided("y", APPROVE, PROMOTED))

    result = stats.compare(promotion_outcomes(a), promotion_outcomes(b), seed=7)

    assert isinstance(result, stats.Refusal)
    assert result.reason == stats.MISSING_OUTCOMES


def test_two_batches_that_approved_one_candidate_twice_refuse_to_be_keyed_by_it():
    """A case id with two answers cannot key a paired comparison, so the batch is refused here."""
    with pytest.raises(ValueError, match="x"):
        promotion_outcomes((_decided("x", APPROVE, PROMOTED), _decided("x", APPROVE, REFUSED)))


# ── purity, structurally ──────────────────────────────────────────────────────────────────
def _imports(source: Path) -> set[str]:
    tree = ast.parse(source.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


def test_the_decide_scorer_reaches_no_io_no_clock_and_no_seeded_draw():
    """The house rule for a pure module: every number is a function of the values handed in, so a
    fixture reproduces it with a pencil."""
    assert _imports(SCORER_SOURCE) <= {
        "__future__",
        "collections",
        "dataclasses",
        "enum",
        "types",
        "typing",
    }
    text = SCORER_SOURCE.read_text()
    for forbidden in ("datetime.now", "utcnow", "time(", "open(", "Path(", "random", "os."):
        assert forbidden not in text, forbidden


def test_the_decide_scorer_restates_no_statistics_and_imports_no_engine_module():
    """The comparison surface is :func:`noctis.eval.stats.compare` itself, handed this module's
    per-case outcomes — so there is no second implementation of a paired test to drift."""
    tree = ast.parse(SCORER_SOURCE.read_text())
    reached = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("noctis")
    }

    assert reached == set()
