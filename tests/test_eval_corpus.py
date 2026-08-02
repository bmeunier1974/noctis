"""The corpus (#197): one frozen stratified split, one content hash, one committed scaffold.

A benchmark that re-deals its tuning/holdout split between two runs is a benchmark whose two
numbers were never comparable, and a corpus that cannot say what it contained is a number nobody
can reproduce. This suite holds both to their contract: the split is computed once from the cases
themselves — same cases, same halves, in any load order and in any process — and the digest is a
function of the corpus content alone, so editing one case moves it and touching nothing reproduces
it byte for byte.

The last two sections are the parts that are not arithmetic: the committed ``cases/`` scaffold (a
README documenting the file format, beside a deny-by-default ignore rule so an operator's mined
corpus never reaches git) and the structural purity check the pure modules in this layer all carry.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from noctis.eval.case import Case, Provenance, ProvenanceKind, Split
from noctis.eval.case_provider import YamlCaseProvider
from noctis.eval.corpus import HOLDOUT_SHARE, Corpus, CorpusError, CorpusIdentity

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_SOURCE = REPO_ROOT / "src/noctis/eval/corpus.py"
CASES_ROOT = REPO_ROOT / "cases"
CASES_README = CASES_ROOT / "README.md"

AUTHORED = Provenance(kind=ProvenanceKind.AUTHORED, reference="2026-07-20")


def _case(case_id: str, **overrides: Any) -> Case:
    """One well-formed case; keyword arguments replace a field each."""
    fields: dict[str, Any] = {
        "case_id": case_id,
        "site_id": "coder",
        "payload": {"brief": f"a strategy for {case_id}"},
        "provenance": AUTHORED,
    }
    fields.update(overrides)
    return Case(**fields)


def _cases(count: int, prefix: str = "case", **overrides: Any) -> tuple[Case, ...]:
    """``count`` cases sharing one stratum, named so the ids sort in generation order."""
    return tuple(_case(f"{prefix}-{index:03d}", **overrides) for index in range(count))


def _splits(corpus: Corpus) -> dict[str, str]:
    """The corpus's assignment as a plain id → split map, for comparing two constructions."""
    return {case.case_id: case.split.value for case in corpus.cases if case.split is not None}


# ── the split is decided once, and the same way every time ────────────────────────────────


def test_the_same_cases_yield_the_same_split_on_every_construction():
    cases = _cases(20)

    assert _splits(Corpus(site_id="coder", cases=cases)) == _splits(
        Corpus(site_id="coder", cases=cases)
    )


def test_a_permuted_load_order_yields_the_same_split_for_every_case():
    """Two providers that disagree about file order still describe one corpus."""
    cases = _cases(20)

    forwards = Corpus(site_id="coder", cases=cases)
    backwards = Corpus(site_id="coder", cases=tuple(reversed(cases)))

    assert _splits(forwards) == _splits(backwards)


def test_the_corpus_holds_its_cases_in_case_id_order_whatever_order_they_arrived_in():
    cases = (_case("gamma"), _case("alpha"), _case("beta"))

    corpus = Corpus(site_id="coder", cases=cases)

    assert [case.case_id for case in corpus.cases] == ["alpha", "beta", "gamma"]


def test_every_case_of_a_constructed_corpus_carries_a_frozen_split():
    corpus = Corpus(site_id="coder", cases=_cases(9))

    assert all(case.split is not None for case in corpus.cases)


def test_the_two_halves_are_disjoint_and_together_are_the_whole_corpus():
    corpus = Corpus(site_id="coder", cases=_cases(13))

    tuning = {case.case_id for case in corpus.tuning}
    holdout = {case.case_id for case in corpus.holdout}

    assert not tuning & holdout
    assert tuning | holdout == {case.case_id for case in corpus.cases}


def test_a_corpus_instance_refuses_field_assignment():
    corpus = Corpus(site_id="coder", cases=_cases(3))

    with pytest.raises(dataclasses.FrozenInstanceError):
        corpus.site_id = "decide"


# ── determinism that survives a fresh interpreter ─────────────────────────────────────────

_PROBE = """
import json
from noctis.eval.case import Case, Provenance, ProvenanceKind
from noctis.eval.corpus import Corpus

cases = tuple(
    Case(
        case_id=f"case-{index:03d}",
        site_id="coder",
        payload={"brief": f"brief {index}"},
        provenance=Provenance(ProvenanceKind.AUTHORED, "2026-07-20"),
        difficulty={"reasoning": "hard" if index % 2 else "easy"},
    )
    for index in range(20)
)
corpus = Corpus(site_id="coder", cases=cases)
print(
    json.dumps(
        {
            "digest": corpus.digest,
            "split": {case.case_id: case.split.value for case in corpus.cases},
        }
    )
)
"""


def _probe(hash_seed: str) -> dict[str, Any]:
    """Build one fixed corpus in a fresh interpreter under ``PYTHONHASHSEED``."""
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(json.loads(completed.stdout))


def test_the_split_is_the_same_in_two_processes_with_different_hash_seeds():
    """No set iteration, no ``hash()``, no seed: a benchmark spans machines and weeks."""
    assert _probe("0")["split"] == _probe("12345")["split"]


def test_the_digest_is_the_same_in_two_processes_with_different_hash_seeds():
    assert _probe("0")["digest"] == _probe("12345")["digest"]


# ── stratification, and the ~70/30 cut ────────────────────────────────────────────────────


def test_a_stratum_of_ten_cases_splits_seven_tuning_and_three_holdout():
    corpus = Corpus(site_id="coder", cases=_cases(10, difficulty={"reasoning": "hard"}))

    assert (len(corpus.tuning), len(corpus.holdout)) == (7, 3)


def test_cases_declaring_no_difficulty_axes_form_one_stratum_split_seven_to_three():
    corpus = Corpus(site_id="coder", cases=_cases(10))

    assert (len(corpus.tuning), len(corpus.holdout)) == (7, 3)


def test_each_difficulty_stratum_is_cut_seven_to_three_of_its_own_cases():
    """The point of stratifying: hard cases are 30% of the holdout, not 30% by luck."""
    hard = _cases(10, prefix="hard", difficulty={"reasoning": "hard"})
    easy = _cases(10, prefix="easy", difficulty={"reasoning": "easy"})

    corpus = Corpus(site_id="coder", cases=hard + easy)

    held_out = [case.case_id for case in corpus.holdout]
    assert sum(1 for case_id in held_out if case_id.startswith("hard")) == 3
    assert sum(1 for case_id in held_out if case_id.startswith("easy")) == 3
    assert (len(corpus.tuning), len(corpus.holdout)) == (14, 6)


def test_identical_difficulty_axes_share_a_stratum_whatever_order_they_were_written_in():
    """A stratum is the axis mapping, not the mapping's insertion order."""
    one = _cases(5, prefix="one", difficulty={"reasoning": "hard", "novelty": "low"})
    other = _cases(5, prefix="other", difficulty={"novelty": "low", "reasoning": "hard"})

    corpus = Corpus(site_id="coder", cases=one + other)

    assert len(corpus.holdout) == 3  # one stratum of ten, not two strata of five


def test_a_stratum_of_one_case_lands_in_tuning_so_the_smallest_corpus_is_iterable_on():
    corpus = Corpus(site_id="coder", cases=_cases(1))

    assert (len(corpus.tuning), len(corpus.holdout)) == (1, 0)


@pytest.mark.parametrize(
    ("size", "holdout"), [(1, 0), (2, 1), (3, 1), (4, 1), (5, 2), (8, 2), (10, 3), (15, 5)]
)
def test_the_holdout_of_a_stratum_is_its_size_times_the_share_rounded_half_up(size, holdout):
    corpus = Corpus(site_id="coder", cases=_cases(size))

    assert len(corpus.holdout) == holdout


def test_the_declared_holdout_share_is_the_exact_ratio_the_cut_uses():
    assert (HOLDOUT_SHARE.numerator, HOLDOUT_SHARE.denominator) == (3, 10)


def test_an_empty_corpus_has_two_empty_halves_and_a_case_count_of_zero():
    corpus = Corpus(site_id="coder")

    assert (corpus.tuning, corpus.holdout, len(corpus)) == ((), (), 0)


# ── a frozen split is never re-dealt ──────────────────────────────────────────────────────


def test_a_case_that_arrives_already_assigned_keeps_the_split_it_arrived_with():
    declared = _cases(9) + (_case("pinned", split=Split.HOLDOUT),)

    corpus = Corpus(site_id="coder", cases=declared)

    assert _splits(corpus)["pinned"] == "holdout"


def test_an_already_frozen_holdout_case_counts_toward_its_stratum_quota():
    """Three cases pinned to the holdout of a ten-case stratum: the other seven are tuning."""
    pinned = tuple(_case(f"pinned-{index}", split=Split.HOLDOUT) for index in range(3))

    corpus = Corpus(site_id="coder", cases=_cases(7) + pinned)

    assert (len(corpus.tuning), len(corpus.holdout)) == (7, 3)


def test_rebuilding_a_corpus_from_its_own_assigned_cases_re_deals_nothing():
    """A re-load of an assigned corpus is the same corpus, not a second deal."""
    first = Corpus(site_id="coder", cases=_cases(20))

    second = Corpus(site_id="coder", cases=first.cases)

    assert _splits(second) == _splits(first)
    assert second.digest == first.digest


def test_a_fully_assigned_corpus_is_reproduced_from_the_splits_its_files_declare():
    declared = tuple(
        _case(f"case-{index}", split=Split.HOLDOUT if index < 3 else Split.TUNING)
        for index in range(10)
    )

    corpus = Corpus(site_id="coder", cases=declared)

    assert {case.case_id for case in corpus.holdout} == {"case-0", "case-1", "case-2"}


# ── the corpus digest ─────────────────────────────────────────────────────────────────────


def test_an_untouched_corpus_reproduces_its_digest():
    cases = _cases(12)

    assert (
        Corpus(site_id="coder", cases=cases).digest == Corpus(site_id="coder", cases=cases).digest
    )


def test_the_digest_does_not_depend_on_the_order_the_cases_were_loaded_in():
    cases = _cases(12)

    assert (
        Corpus(site_id="coder", cases=tuple(reversed(cases))).digest
        == Corpus(site_id="coder", cases=cases).digest
    )


@pytest.mark.parametrize(
    "edit",
    [
        {"payload": {"brief": "something else entirely"}},
        {"tags": ("reversion",)},
        {"difficulty": {"reasoning": "easy"}},
        {"provenance": Provenance(kind=ProvenanceKind.AUTHORED, reference="2026-07-21")},
        {"case_id": "renamed"},
    ],
)
def test_editing_any_part_of_one_case_changes_the_corpus_digest(edit):
    """The digest answers "were these the same asks?" — every field of a case is an ask."""
    before = _cases(6)
    after = before[:-1] + (dataclasses.replace(before[-1], **edit),)

    assert (
        Corpus(site_id="coder", cases=after).digest != Corpus(site_id="coder", cases=before).digest
    )


def test_adding_a_case_changes_the_digest_and_the_case_count():
    before = Corpus(site_id="coder", cases=_cases(6))

    after = Corpus(site_id="coder", cases=_cases(6) + (_case("extra"),))

    assert after.digest != before.digest
    assert (len(after), len(before)) == (7, 6)


def test_two_sites_holding_the_same_asks_have_different_digests():
    cases = _cases(4)
    other = tuple(dataclasses.replace(case, site_id="decide") for case in cases)

    assert (
        Corpus(site_id="decide", cases=other).digest != Corpus(site_id="coder", cases=cases).digest
    )


def test_the_digest_is_a_short_hexadecimal_content_hash():
    digest = Corpus(site_id="coder", cases=_cases(4)).digest

    assert len(digest) == 16
    assert all(character in "0123456789abcdef" for character in digest)


# ── the identity a record folds into its comparable key ───────────────────────────────────


def test_the_corpus_identity_is_the_site_the_digest_and_the_case_count():
    corpus = Corpus(site_id="coder", cases=_cases(5))

    assert corpus.identity == CorpusIdentity(
        site_id="coder", corpus_digest=corpus.digest, case_count=5
    )


def test_the_corpus_identity_renders_as_one_greppable_key_a_record_can_carry():
    corpus = Corpus(site_id="coder", cases=_cases(5))

    assert str(corpus.identity) == f"coder|{corpus.digest}|5"


# ── refusals ──────────────────────────────────────────────────────────────────────────────


def test_two_cases_sharing_one_case_id_are_refused_naming_the_id():
    with pytest.raises(CorpusError) as refusal:
        Corpus(site_id="coder", cases=(_case("twice"), _case("twice"), _case("once")))

    assert "twice" in str(refusal.value)


def test_a_case_belonging_to_another_site_is_refused_naming_both_sites():
    with pytest.raises(CorpusError) as refusal:
        Corpus(site_id="coder", cases=(_case("stray", site_id="decide"),))

    assert "coder" in str(refusal.value) and "decide" in str(refusal.value)


# ── over a corpus a provider actually loaded ──────────────────────────────────────────────


def test_a_corpus_built_from_the_yaml_provider_freezes_a_split_for_every_loaded_case(tmp_path):
    directory = tmp_path / "coder"
    directory.mkdir(parents=True)
    for index in range(10):
        document = {
            "site_id": "coder",
            "payload": {"brief": f"brief {index}"},
            "provenance": "authored:2026-07-20",
            "difficulty": {"reasoning": "hard"},
        }
        (directory / f"case-{index}.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")

    corpus = Corpus(site_id="coder", cases=YamlCaseProvider(cases_root=tmp_path).load("coder"))

    assert (len(corpus.tuning), len(corpus.holdout)) == (7, 3)


# ── the committed scaffold ────────────────────────────────────────────────────────────────


def test_the_committed_cases_scaffold_ships_a_readme():
    assert CASES_README.is_file()


def test_the_cases_readme_documents_every_key_a_case_file_may_declare():
    text = CASES_README.read_text(encoding="utf-8")

    assert all(
        key in text for key in ("site_id", "payload", "provenance", "tags", "difficulty", "split")
    )


def test_the_cases_readme_documents_both_provenance_forms():
    text = CASES_README.read_text(encoding="utf-8")

    assert "mined:" in text and "authored:" in text


def test_the_cases_readme_states_that_a_case_carries_no_expected_output():
    text = CASES_README.read_text(encoding="utf-8")

    assert "expected" in text and "oracle" in text


def test_the_cases_readme_documents_the_per_site_directory_convention():
    text = CASES_README.read_text(encoding="utf-8")

    assert "cases/<site_id>/" in text


def test_the_cases_readme_documents_the_committed_and_gitignored_buckets():
    text = CASES_README.read_text(encoding="utf-8")

    assert "gitignored" in text and "committed" in text


def test_a_case_file_dropped_into_the_cases_tree_is_gitignored_by_default():
    """Deny-by-default, the mandate scaffold's shape: a mined corpus stays the operator's."""
    checked = subprocess.run(
        ["git", "check-ignore", "-q", "cases/coder/mined-case.yaml"], cwd=REPO_ROOT
    )

    assert checked.returncode == 0


def test_the_only_case_path_tracked_in_git_is_the_scaffold_readme():
    tracked = subprocess.run(
        ["git", "ls-files", "cases"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()

    assert tracked == ["cases/README.md"]


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


def test_the_corpus_reaches_no_file_no_clock_and_no_seed():
    """The corpus is built from cases a provider already loaded; nothing here reads a directory."""
    assert _imports(CORPUS_SOURCE) <= {
        "__future__",
        "collections",
        "dataclasses",
        "fractions",
        "hashlib",
        "json",
        "types",
        "typing",
        "noctis",
    }
    text = CORPUS_SOURCE.read_text()
    for forbidden in ("open(", "Path(", "os.", "import random", "datetime.now", "utcnow"):
        assert forbidden not in text, forbidden
