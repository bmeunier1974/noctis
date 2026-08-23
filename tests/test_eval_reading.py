"""The eval layer's reading vocabulary (#303): one ``fmt``, one ``table``, two pair manifests.

Three properties carry this suite, and they are the reason the module exists at all:

* **One spelling of an absence.** ``fmt`` is the single place a ``None`` becomes the word ``n/a``,
  a rate becomes four decimals and a flag becomes ``yes``/``no``. Every reading in the eval layer
  formats through it, so two blocks printed by two modules cannot disagree about what a missing
  number looks like.
* **A co-primary pair is declared once.** A :class:`~noctis.eval.reading.PairManifest` names a
  pair's figures, the order a block publishes them in, the order a report prints them in, the
  flagship figure that may never appear alone, and the refusal for when it does. The owning
  dataclass keeps its arithmetic and routes its rendering through the manifest, so a row cannot be
  restated in three modules and drift in two of them.
* **The module is pure, and provably so.** It imports the standard library and nothing else — not
  the engine, not another eval module, no file and no clock — which is what lets every site's
  reading depend on it without dragging anything behind it. Both the static import walk and a
  fresh-interpreter module closure say so.

Beside them sit the site-neutral arithmetic the module has grown since: ``fold_by_case`` and
``strict_majority``, which settle what a case's reps contribute (#305), and ``strata``, the one
grouping loop both sites' readings are stratified by (#306). Each is generic over a key or a pair
of callables — which is exactly what lets it live here rather than twice, at a site.

The byte-for-byte pins against ``ApprovalPair.render()`` and ``PassRates.render()`` are the move's
own safety net, beside the golden records in ``tests/test_eval_goldens.py``.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import pytest

from noctis.eval import reading
from noctis.eval.coder_scorer import (
    FEEDBACK_LABEL,
    PASS_LABEL_KEY,
    PassRates,
    coder_block,
    score_coder_jobs,
)
from noctis.eval.coder_site import AttemptRecord, JobRecord
from noctis.eval.decide_scorer import ApprovalPair
from noctis.eval.decide_site import pair_block
from noctis.eval.reading import (
    APPROVAL_PAIR,
    NOT_APPLICABLE,
    PAIR_MANIFESTS,
    PASS_RATES,
    PairManifest,
    PairRow,
    fmt,
    fold_by_case,
    strata,
    strict_majority,
    table,
)

READING_SOURCE = Path(str(reading.__file__))

PAIR = ApprovalPair(
    agreement=0.5,
    approval_rate=0.25,
    decided=8,
    approvals=2,
    labeled_approvals=2,
    unlabeled_approvals=0,
    promoted=1,
)

RATES = PassRates(
    first_attempt_pass_rate=0.5,
    job_pass_rate=0.75,
    cases=4,
    jobs=4,
    first_attempt_passes=2,
    passed_jobs=3,
    unattempted_jobs=1,
)

#: One scored coder batch, so the manifest's block can be held against the published one. The
#: arithmetic is ``tests/test_eval_coder_scorer.py``'s business; here it is only a source of rates.
CODER_METRICS = score_coder_jobs(
    [
        JobRecord(
            case_id="a",
            strategy="momentum_a",
            passed=True,
            attempts=(
                AttemptRecord(attempt=1, passed=False, error="gate refused"),
                AttemptRecord(attempt=2, passed=True),
            ),
        )
    ]
)


# ── the shared vocabulary: one spelling, wherever a record says the word ──────────────────
#: Every word the eval layer's records carry that more than one module has to say (#305).
SHARED_SPELLINGS = frozenset(
    {"n/a", "fresh", "recorded", "strata", "retrospective", "answers", "attempt_calls"}
)


def _declared_spellings(source: Path) -> list[str]:
    """Every shared word a module binds a module-level name to — a spelling of its own, listed."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    return [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and node.value.value in SHARED_SPELLINGS
    ]


def test_the_shared_spellings_are_the_words_a_published_reading_carries() -> None:
    assert (reading.NOT_APPLICABLE, reading.ANSWERS_FRESH, reading.ANSWERS_RECORDED) == (
        "n/a",
        "fresh",
        "recorded",
    )


def test_the_headline_keys_are_the_three_facts_a_dials_block_states_up_front() -> None:
    assert (reading.RETROSPECTIVE_KEY, reading.ANSWERS_KEY, reading.ATTEMPT_CALLS_KEY) == (
        "retrospective",
        "answers",
        "attempt_calls",
    )


def test_the_per_axis_breakdown_rides_under_the_one_key_both_sites_spell_the_same_way() -> None:
    assert reading.STRATA_KEY == "strata"


def test_this_module_declares_every_shared_spelling_exactly_once() -> None:
    """A list, not a set: a second copy *here* would be the same duplication, one file smaller."""
    assert sorted(_declared_spellings(READING_SOURCE)) == sorted(SHARED_SPELLINGS)


def test_no_other_eval_module_declares_a_shared_spelling_of_its_own() -> None:
    """The words a record carries are imported or re-exported from here — never restated (#305)."""
    offenders = {
        source.name: declared
        for source in sorted(READING_SOURCE.parent.glob("*.py"))
        if source != READING_SOURCE and (declared := _declared_spellings(source))
    }

    assert offenders == {}


# ── fmt: the one place an absence becomes a word ──────────────────────────────────────────
def test_a_missing_figure_renders_as_the_literal_n_a() -> None:
    assert fmt(None) == "n/a"


def test_a_rate_renders_at_four_decimals() -> None:
    assert fmt(0.5) == "0.5000"
    assert fmt(1.0) == "1.0000"


def test_a_flag_renders_as_yes_or_no_rather_than_a_python_repr() -> None:
    assert fmt(True) == "yes"
    assert fmt(False) == "no"


def test_a_count_and_a_word_render_verbatim() -> None:
    assert fmt(3) == "3"
    assert fmt("fresh") == "fresh"


def test_the_flag_arm_wins_over_the_number_arms_because_a_bool_is_an_int() -> None:
    """``isinstance(True, int)`` is the trap: without the flag arm first, a flag prints ``True``."""
    assert fmt(False) != "0"
    assert fmt(True) != "1"


# ── table: two columns, at the widths a caller asks for ───────────────────────────────────
def test_a_row_is_the_label_left_padded_and_the_value_right_aligned_beside_it() -> None:
    assert table([("Approval rate", 0.25)]) == f"{'Approval rate':<28}{'0.2500':>12}"


def test_the_default_widths_are_twenty_eight_and_twelve() -> None:
    (line,) = table([("Label", 1)]).splitlines()

    assert len(line) == 40


def test_a_caller_may_widen_the_label_column() -> None:
    (line,) = table([("Label", 1)], label_w=36).splitlines()

    assert len(line) == 48


def test_a_caller_may_widen_the_value_column() -> None:
    (line,) = table([("Label", 1)], value_w=22).splitlines()

    assert len(line) == 50


def test_every_row_is_one_line_in_the_order_it_was_handed_in() -> None:
    rendered = table([("First", 1), ("Second", 2)])

    assert [line.strip().split()[0] for line in rendered.splitlines()] == ["First", "Second"]


def test_a_table_of_no_rows_is_the_empty_string() -> None:
    assert table([]) == ""


def test_a_missing_cell_becomes_n_a_because_the_table_formats_through_fmt() -> None:
    assert table([("Agreement", None)]).endswith("n/a")


def test_the_word_n_a_is_spelled_in_exactly_one_place_in_the_reading_module() -> None:
    """The whole point of a shared vocabulary: one literal, not one per renderer."""
    tree = ast.parse(READING_SOURCE.read_text(encoding="utf-8"))
    literals = [
        node for node in ast.walk(tree) if isinstance(node, ast.Constant) and node.value == "n/a"
    ]

    assert len(literals) == 1


# ── the two manifests: declared once, rendered by the type that owns the arithmetic ───────
def test_both_pair_manifests_are_exported_together() -> None:
    assert PAIR_MANIFESTS == (APPROVAL_PAIR, PASS_RATES)


def test_the_approval_manifest_names_the_keys_the_approval_block_publishes_in_order() -> None:
    assert APPROVAL_PAIR.keys == (
        "agreement",
        "approval_rate",
        "decided",
        "approvals",
        "labeled_approvals",
        "unlabeled_approvals",
        "promoted",
    )


def test_the_pass_rate_manifest_names_the_keys_the_rates_block_publishes_in_order() -> None:
    assert PASS_RATES.keys == (
        "pass_label",
        "first_attempt_pass_rate",
        "job_pass_rate",
        "first_attempt_passes",
        "passed_jobs",
    )


def test_the_pass_rate_manifest_spells_the_label_key_the_coder_reading_publishes() -> None:
    assert PASS_LABEL_KEY in PASS_RATES.keys


def test_the_approval_manifest_renders_exactly_what_the_pair_itself_renders() -> None:
    assert APPROVAL_PAIR.render(PAIR) == PAIR.render()


def test_the_pass_rate_manifest_renders_exactly_what_the_rates_themselves_render() -> None:
    assert PASS_RATES.render(RATES) == RATES.render()


def test_the_approval_manifest_builds_exactly_the_block_the_decide_reading_publishes() -> None:
    published = APPROVAL_PAIR.block(PAIR)

    assert published == pair_block(PAIR)
    assert list(published) == list(pair_block(PAIR))


def test_the_pass_rate_manifest_builds_exactly_the_block_the_coder_reading_publishes() -> None:
    published = PASS_RATES.block(RATES)
    inside = coder_block(CODER_METRICS)["rates"]

    assert list(published) == list(inside)
    assert published["pass_label"] == FEEDBACK_LABEL


def test_the_job_pass_rate_row_carries_the_feedback_label_the_pair_was_told_to_wear() -> None:
    """The qualification travels in the printed label, filled from the pair's own field."""
    labels = [label for label, _ in PASS_RATES.rows(RATES)]

    assert f"Job pass rate ({FEEDBACK_LABEL})" in labels


# ── the flagship, and the refusal for when it turns up alone ──────────────────────────────
def test_the_approval_manifest_names_agreement_as_the_figure_that_may_never_stand_alone() -> None:
    assert APPROVAL_PAIR.flagship == "agreement"


def test_the_pass_rate_manifest_names_the_retry_informed_rate_as_its_flagship() -> None:
    assert PASS_RATES.flagship == "job_pass_rate"


def test_every_manifests_flagship_is_one_of_the_keys_it_publishes() -> None:
    for manifest in PAIR_MANIFESTS:
        assert manifest.flagship in manifest.keys, manifest.name


def test_every_manifests_refusal_names_the_flagship_it_is_refusing_to_print_alone() -> None:
    for manifest in PAIR_MANIFESTS:
        assert manifest.flagship in manifest.refusal, manifest.name
        assert manifest.refusal.startswith("REFUSED"), manifest.name


# ── the D2 guard: a manifest may only name figures its pair really carries ─────────────────
def test_the_approval_manifest_reads_only_fields_the_approval_pair_carries() -> None:
    assert APPROVAL_PAIR.attributes() <= {field.name for field in fields(ApprovalPair)}


def test_the_pass_rate_manifest_reads_only_fields_the_pass_rates_carry() -> None:
    assert PASS_RATES.attributes() <= {field.name for field in fields(PassRates)}


def test_a_manifest_that_publishes_a_key_no_figure_declares_is_refused() -> None:
    with pytest.raises(ValueError, match="ghost"):
        PairManifest(
            name="broken",
            figures=(PairRow("agreement", "Agreement"),),
            keys=("agreement", "ghost"),
            flagship="agreement",
            refusal="REFUSED — agreement alone.",
        )


def test_a_manifest_whose_flagship_is_not_one_of_its_figures_is_refused() -> None:
    with pytest.raises(ValueError, match="phantom"):
        PairManifest(
            name="broken",
            figures=(PairRow("agreement", "Agreement"),),
            keys=("agreement",),
            flagship="phantom",
            refusal="REFUSED — phantom alone.",
        )


# ── strict_majority: more than half of them said it, or nobody did ────────────────────────
@dataclass(frozen=True)
class _Answer:
    """A stand-in for whatever a site folds: the function knows only the key it is handed."""

    who: str
    verdict: str


def _verdict(answer: _Answer) -> str:
    return answer.verdict


def test_a_verdict_more_than_half_the_answers_hold_is_the_majority() -> None:
    answers = (_Answer("a", "approve"), _Answer("b", "approve"), _Answer("c", "reject"))

    assert strict_majority(answers, key=_verdict) == _Answer("a", "approve")


def test_the_majority_is_the_first_answer_holding_it_so_a_caller_reads_a_real_one() -> None:
    answers = (
        _Answer("a", "reject"),
        _Answer("b", "approve"),
        _Answer("c", "approve"),
        _Answer("d", "approve"),
    )

    assert strict_majority(answers, key=_verdict) == _Answer("b", "approve")


def test_an_exact_tie_holds_no_majority_and_none_is_invented_for_it() -> None:
    answers = (_Answer("a", "approve"), _Answer("b", "reject"))

    assert strict_majority(answers, key=_verdict) is None


def test_a_plurality_short_of_half_holds_no_majority_either() -> None:
    answers = (
        _Answer("a", "approve"),
        _Answer("b", "approve"),
        _Answer("c", "reject"),
        _Answer("d", "revise"),
    )

    assert strict_majority(answers, key=_verdict) is None


def test_one_answer_is_its_own_majority() -> None:
    assert strict_majority((_Answer("a", "approve"),), key=_verdict) == _Answer("a", "approve")


def test_no_answers_hold_no_majority() -> None:
    assert strict_majority((), key=_verdict) is None


# ── fold_by_case: one case, one group, in case-id order ───────────────────────────────────
@dataclass(frozen=True)
class _Rep:
    """One rep's answer to one case — all ``fold_by_case`` reads of it is the case it answered."""

    case_id: str
    rep: int


def _case_id(one: _Rep) -> str:
    return one.case_id


def test_every_answer_to_one_case_folds_into_that_cases_group() -> None:
    answered = (_Rep("a", 1), _Rep("b", 1), _Rep("a", 2))

    assert fold_by_case(answered, key=_case_id) == (
        (_Rep("a", 1), _Rep("a", 2)),
        (_Rep("b", 1),),
    )


def test_the_groups_come_in_case_id_order_whatever_order_the_answers_arrived_in() -> None:
    answered = (_Rep("c", 1), _Rep("a", 1), _Rep("b", 1))

    assert tuple(group[0].case_id for group in fold_by_case(answered, key=_case_id)) == (
        "a",
        "b",
        "c",
    )


def test_one_group_per_case_id_however_many_reps_answered_it() -> None:
    answered = (_Rep("a", 1), _Rep("a", 2), _Rep("a", 3), _Rep("b", 1))

    assert tuple(len(group) for group in fold_by_case(answered, key=_case_id)) == (3, 1)


def test_answers_keep_the_order_they_arrived_in_inside_their_own_group() -> None:
    answered = (_Rep("a", 3), _Rep("a", 1), _Rep("a", 2))

    assert tuple(one.rep for one in fold_by_case(answered, key=_case_id)[0]) == (3, 1, 2)


def test_no_answers_fold_into_no_groups() -> None:
    assert fold_by_case((), key=_case_id) == ()


# ── strata: one grouping loop, whatever a site stratifies ─────────────────────────────────
@dataclass(frozen=True)
class _Labelled:
    """A stand-in for whatever a site stratifies — a coder job record, a DECIDE outcome."""

    who: str
    difficulty: dict[str, str]


def _level(one: _Labelled, axis: str) -> str | None:
    """How a caller reads a level off its own item: the label it carries, or nothing."""
    return one.difficulty.get(axis)


def _who(items: Sequence[_Labelled]) -> dict[str, object]:
    """A stand-in for a site's own block builder — the names in this level, and how many."""
    return {"count": len(items), "who": [one.who for one in items]}


AXES = ("margin", "surface")

LABELLED = (
    _Labelled("a", {"margin": "near", "surface": "exits"}),
    _Labelled("b", {"margin": "comfortable", "surface": "exits"}),
    _Labelled("c", {"margin": "near"}),
)


def test_every_declared_axis_is_stratified_in_the_order_it_was_declared() -> None:
    assert list(strata(AXES, LABELLED, level_of=_level, block_of=_who)) == ["margin", "surface"]


def test_an_axis_carries_the_levels_its_items_were_labelled_on_in_sorted_order() -> None:
    assert list(strata(AXES, LABELLED, level_of=_level, block_of=_who)["margin"]) == [
        "comfortable",
        "near",
    ]


def test_a_levels_block_is_built_over_exactly_the_items_that_carry_that_level() -> None:
    stratified = strata(AXES, LABELLED, level_of=_level, block_of=_who)

    assert stratified["margin"]["near"] == {"count": 2, "who": ["a", "c"]}


def test_an_item_carrying_no_level_on_an_axis_is_stratified_under_the_one_word_for_an_absence() -> (
    None
):
    stratified = strata(AXES, LABELLED, level_of=_level, block_of=_who)

    assert stratified["surface"][NOT_APPLICABLE] == {"count": 1, "who": ["c"]}


def test_a_level_nobody_carries_is_absent_rather_than_present_at_zero() -> None:
    """Nothing was measured there, and an empty row would read as a measurement."""
    stratified = strata(AXES, LABELLED, level_of=_level, block_of=_who)

    assert "scale_free" not in stratified["surface"]
    assert set(stratified["surface"]) == {"exits", NOT_APPLICABLE}


def test_items_keep_the_order_they_arrived_in_inside_their_own_level() -> None:
    shuffled = (LABELLED[2], LABELLED[0])

    assert strata(AXES, shuffled, level_of=_level, block_of=_who)["margin"]["near"]["who"] == [
        "c",
        "a",
    ]


def test_a_site_that_declares_no_axes_stratifies_into_an_empty_mapping() -> None:
    assert strata((), LABELLED, level_of=_level, block_of=_who) == {}


def test_an_axis_nothing_was_measured_on_carries_no_levels_at_all() -> None:
    """No items is not a level of its own: the axis is declared, and its breakdown is empty."""
    assert strata(AXES, (), level_of=_level, block_of=_who) == {"margin": {}, "surface": {}}


# ── purity, structurally ──────────────────────────────────────────────────────────────────
def _import_roots(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_the_reading_module_imports_the_standard_library_and_nothing_else() -> None:
    """Every site's reading depends on this module, so it must drag nothing behind it."""
    roots = _import_roots(READING_SOURCE)

    assert roots
    assert roots <= sys.stdlib_module_names
    assert "noctis" not in roots


def test_the_reading_module_reaches_no_file_no_clock_and_no_seeded_draw() -> None:
    text = READING_SOURCE.read_text(encoding="utf-8")

    for forbidden in ("open(", "Path(", "os.", "random", "datetime", "time."):
        assert forbidden not in text, forbidden


def test_a_fresh_interpreter_that_imports_the_reading_module_loads_no_other_noctis_module() -> None:
    """The static walk cannot see a transitive import; a module closure in a fresh process can."""
    probe = (
        "import json, sys\n"
        "import noctis.eval.reading\n"
        "print(json.dumps(sorted(n for n in sys.modules if n.split('.')[0] == 'noctis')))\n"
    )
    finished = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert json.loads(finished.stdout) == ["noctis", "noctis.eval", "noctis.eval.reading"]
