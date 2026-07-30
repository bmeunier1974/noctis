"""Structured gate evidence beside every promotion decision (story #141, epic #126).

The rejections **are** the product. "47 of 66 candidates died at the symbol-holdout gate" is the
sentence that makes this project's results credible where an equity curve does not — and until now
it could not be computed from anything stored, because a promotion rationale is prose.

So ``decide()`` gains ``gates: tuple[GateResult, ...]``: what each gate measured, against what
threshold, as it evaluated. **No decision moves.** That is not asserted here, it is *proved* — the
corpus in ``tests/_promotion_cases.py`` was snapshotted against the promotion module as it stood
before this change, and the first test below replays every case against the golden.

Everything asserted is external behaviour: the values a decision carries, the order the evidence
arrives in, and — structurally, in a fresh subprocess — that nothing on the promotion path can
reach the reporting package that consumes it.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from noctis.champions import PromotionRules, decide
from noctis.champions.promotion import FINAL_GATES, GATE_ORDER, Decision, GateResult

from ._promotion_cases import card, cases, outcomes, read_golden, rules

SRC = Path(__file__).resolve().parents[1] / "src"

# Every module a promotion decision runs through: the gates themselves, the board that applies
# them, and the scorecard the numbers are read off. The isolation rule below covers all three.
PROMOTION_PATH = (
    "noctis.champions.promotion",
    "noctis.champions.registry",
    "noctis.backtest.scorecard",
)


# ── gate parity: the outcomes are bit-identical to the pre-change golden ────────────────────


def test_every_decision_matches_the_golden_captured_before_gate_evidence_landed():
    """The parity proof. The golden was generated from ``decide()`` *before* it carried gates, so
    a difference here means a judgment moved — an engine change, never a test to update."""
    assert outcomes() == read_golden()


def test_the_corpus_covers_every_gate_including_the_ones_that_pass():
    """A parity golden only proves what it exercises: every named gate must appear as the *failing*
    one in at least one case, or a gate could be rewritten with the corpus none the wiser."""
    failed = {
        result.gate
        for case in cases()
        for result in decide(case.challenger, case.champions, case.rules).gates
        if not result.passed
    }
    assert failed == {*GATE_ORDER, *FINAL_GATES}


def test_existing_callers_that_ignore_the_new_field_keep_working():
    """``Decision`` is still constructible — and comparable — without gates: every caller that
    reads ``promote`` / ``rationale`` / ``demote_index`` is untouched by this story."""
    assert Decision(True, "promoted: …") == Decision(True, "promoted: …", gates=())
    decision = decide(card(test=1.5), [], rules())
    assert (decision.promote, decision.demote_index) == (True, None)


# ── the evidence itself ─────────────────────────────────────────────────────────────────────


def _gates(challenger, champions=(), **rule_overrides) -> dict[str, GateResult]:
    return {
        result.gate: result
        for result in decide(challenger, list(champions), rules(**rule_overrides)).gates
    }


def test_a_promotion_carries_every_gate_it_cleared_with_what_each_one_measured():
    decision = decide(
        card(test=1.4, train=1.7, holdout=0.8, symbol_holdout=0.6, exposures=(1.0, 1.0, 0.0, 1.0)),
        [],
        rules(min_test_activity=0.5, min_holdout_metric=0.1, min_symbol_holdout_metric=0.1),
    )

    assert decision.promote is True
    assert [result.gate for result in decision.gates] == [*GATE_ORDER, "minimum_bar"]
    assert all(result.passed for result in decision.gates)
    measured = {result.gate: (result.observed, result.threshold) for result in decision.gates}
    assert measured["activity_floor"] == (0.75, 0.5)
    assert measured["overfit_gap"] == (pytest.approx(0.3), 1.0)
    assert measured["forward_holdout"] == (0.8, 0.1)
    assert measured["symbol_holdout"] == (0.6, 0.1)
    assert measured["minimum_bar"] == (1.4, 0.0)


def test_a_rejection_short_circuits_so_the_gates_are_the_ones_reached_plus_the_failure():
    """The honest shape: a rejected candidate was never measured against the gates *after* the one
    that stopped it, so the record must not pretend it was."""
    decision = decide(card(test=1.0, holdout=-0.5, symbol_holdout=-9.0), [], rules())

    assert decision.promote is False
    assert [result.gate for result in decision.gates] == [
        "validated",
        "activity_floor",
        "overfit_gap",
        "reverse_gap",
        "magnitude_cap",
        "forward_holdout",
    ]
    failed = decision.gates[-1]
    assert (failed.gate, failed.passed, failed.observed, failed.threshold) == (
        "forward_holdout",
        False,
        -0.5,
        0.0,
    )
    assert "symbol_holdout" not in {result.gate for result in decision.gates}


def test_the_gate_that_failed_is_the_last_one_and_the_only_one_that_failed():
    for case in cases():
        decision = decide(case.challenger, case.champions, case.rules)
        failures = [result for result in decision.gates if not result.passed]
        if decision.promote:
            assert failures == [], case.name
        else:
            assert len(failures) == 1, case.name
            assert failures[0] is decision.gates[-1], case.name


def test_the_gate_order_is_the_documented_one_for_every_case():
    """Activity floor → overfit gap → forward holdout → symbol holdout → consistency → family
    slot → beat the weakest. Every decision's evidence is a prefix of that order, whatever its
    outcome."""
    for case in cases():
        decision = decide(case.challenger, case.champions, case.rules)
        emitted = [result.gate for result in decision.gates]
        head = [name for name in emitted if name not in FINAL_GATES]
        assert head == list(GATE_ORDER[: len(head)]), case.name
        assert set(emitted) - set(head) <= set(FINAL_GATES), case.name


def test_a_card_that_was_never_validated_reports_the_one_gate_it_reached():
    decision = decide(card(stage="prefilter_rejected"), [], rules())

    assert [(g.gate, g.passed) for g in decision.gates] == [("validated", False)]
    assert decision.gates[0].note is not None  # a null observation still says why


def test_a_gate_that_is_switched_off_is_recorded_as_inert_rather_than_omitted():
    """A disabled gate still measured the candidate — it just could not stop it. Recording that
    keeps the funnel's denominators honest: "0 of 66 died at the consistency gate" is a fact only
    when the gate is known to have been in the path."""
    gates = _gates(card(per_symbol={"AAA": 1.5, "BBB": -0.4}), max_reverse_gap=0.0)

    for name in ("reverse_gap", "magnitude_cap", "symbol_consistency", "activity_floor"):
        assert gates[name].passed is True
        assert gates[name].note is not None and "inert" in gates[name].note
    assert gates["symbol_consistency"].observed == 0.5  # measured anyway, and reported


def test_a_holdout_the_scorecard_never_carried_is_null_rather_than_zero():
    gates = _gates(card(holdout=None, symbol_holdout=None))

    assert gates["forward_holdout"].observed is None
    assert gates["symbol_holdout"].observed is None
    assert "no forward-holdout" in str(gates["forward_holdout"].note)


def test_the_final_gate_names_which_slot_the_challenger_was_judged_for():
    board = [card("champ_a", test=1.0), card("champ_b", test=0.5), card("champ_c", test=2.0)]

    free = _gates(card(test=1.5))
    beat = _gates(card(test=0.9), board)

    assert free["minimum_bar"].observed == 1.5 and "free slot" in str(free["minimum_bar"].note)
    assert (beat["beat_weakest"].observed, beat["beat_weakest"].threshold) == (0.9, 0.5)
    assert "minimum_bar" not in beat


def test_a_family_slot_rejection_still_records_every_honesty_gate_it_cleared():
    """The re-tune is turned away, but *how it scored* is still on the record — which is the whole
    reason the family gate sits after the honesty gates rather than in front of them."""
    decision = decide(
        card("vol_breakout", test=1.4, train=1.7, holdout=0.8, symbol_holdout=0.6),
        [card("vol_breakout", test=1.0)],
        rules(min_holdout_metric=0.1, min_symbol_holdout_metric=0.1),
    )

    assert decision.promote is False
    assert [result.gate for result in decision.gates] == list(GATE_ORDER)
    assert all(result.passed for result in decision.gates[:-1])
    failed = decision.gates[-1]
    assert (failed.gate, failed.passed, failed.observed, failed.threshold) == (
        "family_slot",
        False,
        None,
        None,
    )
    measured = {result.gate: result.observed for result in decision.gates}
    assert measured["forward_holdout"] == 0.8 and measured["symbol_holdout"] == 0.6


def test_the_family_gate_names_the_incumbent_holding_the_slot():
    """``crowned_at`` is evidence, not an input to the verdict: the board hands it over so the
    note can say who holds the slot; a caller with none simply gets a shorter note."""
    board = [card("champ_a", test=1.0), card("vol_breakout", test=0.5)]

    named = decide(card("vol_breakout"), board, rules(), crowned_at=[None, "2026-07-29T01:50:52Z"])
    anonymous = decide(card("vol_breakout"), board, rules())

    assert "2026-07-29T01:50:52Z" in str(named.gates[-1].note)
    assert "'lookback': 40" in str(named.gates[-1].note)  # the incumbent's params
    assert named.rationale == anonymous.rationale
    assert "crowned" not in str(anonymous.gates[-1].note)


def test_a_board_with_no_incumbent_of_the_family_records_the_gate_as_cleared():
    gates = _gates(card("vol_breakout", test=1.5), [card("champ_a", test=1.0)])

    assert gates["family_slot"].passed is True
    assert "vol_breakout" in str(gates["family_slot"].note)


def test_a_stale_board_is_judged_on_the_minimum_bar_and_says_so():
    stale = [card("champ_a", test=1.0, metric_name="sortino"), card("champ_b", test=9.0)]

    gates = _gates(card(test=0.3), stale, champion_count=2)

    assert gates["minimum_bar"].passed is True
    assert "stale" in str(gates["minimum_bar"].note)


# ── the board persists the evidence, so the record can read it back ─────────────────────────


def test_the_champion_board_journals_each_decisions_gates_beside_its_rationale(tmp_path):
    from noctis.champions import ChampionRegistry

    registry = ChampionRegistry(tmp_path / "champions.json", capacity=3)
    registry.consider(card("winner", test=1.5), rules())
    registry.consider(card("loser", test=1.0, holdout=-0.5), rules())

    board = ChampionRegistry(tmp_path / "champions.json", capacity=3)
    journaled = {entry["family"]: entry for entry in board.history}
    assert [gate["gate"] for gate in journaled["loser"]["gates"]][-1] == "forward_holdout"
    assert journaled["loser"]["gates"][-1] == {
        "gate": "forward_holdout",
        "passed": False,
        "observed": -0.5,
        "threshold": 0.0,
        "note": None,
    }
    assert journaled["winner"]["gates"][-1]["gate"] == "minimum_bar"
    assert all(gate["passed"] for gate in journaled["winner"]["gates"])


# ── isolation: the record is evidence, never a gate ─────────────────────────────────────────


def test_the_promotion_path_imports_nothing_from_the_reporting_package():
    """AGENTS.md rule 2 and epic invariant 1, enforced structurally: nothing computed *for* the
    record may enter a decision path.

    The check is deliberately over the whole ``noctis.reporting`` package rather than one module.
    ``reporting/metrics.py`` — the extended performance metrics the epic's isolation criterion
    names — lands in story #142; a test naming it today would pass vacuously and could be deleted
    as dead before it ever guarded anything. A package-wide rule is meaningful now (the record
    builder and the run store already live there), and it is *strictly stronger* the moment #142
    adds the module: an import of ``noctis.reporting.metrics`` fails this by construction.

    Transitive, in a fresh subprocess, because a direct-import check would miss a gate module that
    reached the reporting package through a collaborator.
    """
    probe = (
        "import sys;"
        f"[__import__(name) for name in {PROMOTION_PATH!r}];"
        "print(sorted(m for m in sys.modules if m.startswith('noctis.reporting')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]", (
        "the promotion path pulled in the reporting package: "
        f"{result.stdout.strip()}. The record is evidence, never a gate — a number computed for "
        "reporting (noctis.reporting.metrics, story #142) must never be reachable from decide()."
    )


@pytest.mark.parametrize("module", PROMOTION_PATH)
def test_no_module_on_the_promotion_path_names_the_reporting_package(module: str):
    """The same rule, read off the source: a direct import is the one a review sees."""
    source = SRC / Path(*module.split(".")).with_suffix(".py")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    named = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}

    assert not [name for name in named if name.startswith("noctis.reporting")], source


# ── the canonical prose is held to the ladder it describes ─────────────────────────────────


def _ladder() -> list[tuple[str, str]]:
    """The promotion module docstring's numbered gate ladder: each step's title and its prose."""
    from noctis.champions import promotion

    steps = re.split(r"^\d+\. ", promotion.__doc__ or "", flags=re.MULTILINE)[1:]
    return [(step.split("**")[1], step) for step in steps]


def test_the_module_docstring_ladder_is_the_order_the_gates_run():
    """The ladder is this project's canonical prose about promotion — an operator, the research
    agent's contract sheet and the docs all read it, so a gate that moves in code and not here
    would leave the authoritative description wrong."""
    assert [title for title, _ in _ladder()] == [
        "Activity floor",
        "Gap guard",
        "Forward-holdout gate",
        "Symbol-holdout gate",
        "Consistency gate (optional)",
        "One slot per family",
        "Free slot",
        "Stale champions lose first",
        "Beat the weakest",
    ]


def test_the_docstring_ladder_is_numbered_consecutively():
    from noctis.champions import promotion

    numbered = re.findall(r"^(\d+)\. \*\*", promotion.__doc__ or "", flags=re.MULTILINE)
    assert [int(n) for n in numbered] == list(range(1, len(numbered) + 1))


def test_the_docstring_explains_that_the_family_gate_precedes_stale_displacement():
    """The one interaction a reader cannot infer from the order alone: a stale champion is
    displaceable, but never by its own family's re-tune."""
    stale = dict(_ladder())["Stale champions lose first"]
    assert "family" in stale and "own" in stale


def test_promotion_rules_still_map_from_settings_unchanged():
    """The gates' one config seam is untouched by the evidence they now carry."""
    from noctis.config.settings import Settings

    mapped = PromotionRules.from_settings(Settings(promotion={"max_gap": 0.6}))
    assert mapped.max_gap == 0.6 and mapped.champion_count == 3
