"""The committed coder corpus (#218): a bucket-shaped scaffold, a stamped split, six canaries.

Three things are held to their contract here. **The scaffold**: the coder corpus is the first one
this repo ships to every user, so what is committed (the README, ``edge/``, ``canary/``) and what
stays the operator's (``field/``, ``replay/``) is asserted against git itself rather than against a
comment. **The split**: it is dealt once by an authoring-time tool and *written into the files*, so
the loader reads an assignment rather than recomputing one — a benchmark whose halves can move
between two runs never had two comparable numbers. **The canaries**: six one-rule adaptations of the
simplest committed seed, honestly labelled on all seven axes, each loading through the real provider
and the real coder validator, so a red canary indicts the harness and never the model.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from noctis.eval.case import Case, MalformedCase, Provenance, ProvenanceKind, Split
from noctis.eval.case_provider import MissingCorpus
from noctis.eval.coder_case import AXIS_LEVELS, REQUIRED_BRIEF_KEYS, Axis, Bucket, coder_payload
from noctis.eval.coder_corpus import (
    CODER_SITE_ID,
    COMMITTED_BUCKETS,
    LOCAL_BUCKETS,
    CoderCaseProvider,
    bucket_of,
    deal_splits,
    load_coder_corpus,
    stamp_splits,
)
from noctis.eval.corpus import HOLDOUT_SHARE, Corpus
from noctis.research.author import StrategyBrief

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_ROOT = REPO_ROOT / "cases"
CODER_ROOT = CASES_ROOT / CODER_SITE_ID
CODER_README = CODER_ROOT / "README.md"

# The seed every canary adapts, and the tags that record that lineage in the case itself.
SEED = "sma_crossover"
ADAPTS_TAG = f"adapts:{SEED}"
RULE_TAG_PREFIX = "rule:"

AUTHORED = Provenance(kind=ProvenanceKind.AUTHORED, reference="2026-08-02")

# The easy end of every axis — what a canary is labelled with, and a valid label set for a fixture.
EASY_AXES: Mapping[str, str] = {axis.value: AXIS_LEVELS[axis][0] for axis in Axis}

BRIEF: Mapping[str, str] = {
    "thesis": "A fast moving average above a slow one says demand is outpacing its own average.",
    "entry_exit": "Long while the fast SMA is above the slow SMA; flat otherwise. No shorting.",
    "param_space": "fast 3-30, slow 20-100",
    "scenarios": "trend_ride_then_rollover: long during the trend, flat after the selloff.",
}


def _document(bucket: Bucket, **overrides: Any) -> dict[str, Any]:
    """A complete, coherent coder case document for ``bucket``; keywords replace one key each."""
    document: dict[str, Any] = {
        "site_id": CODER_SITE_ID,
        "payload": dict(BRIEF),
        "provenance": str(AUTHORED),
        "tags": [f"bucket:{bucket.value}"],
        "difficulty": dict(EASY_AXES),
    }
    document.update(overrides)
    return document


def _write(root: Path, bucket: Bucket, name: str, **overrides: Any) -> Path:
    """Write one case into ``<root>/coder/<bucket>/<name>.yaml``, the layout the loader walks."""
    directory = root / CODER_SITE_ID / bucket.value
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(_document(bucket, **overrides), sort_keys=True), "utf-8")
    return path


def _case(case_id: str, bucket: Bucket = Bucket.CANARY, **overrides: Any) -> Case:
    """One in-memory coder case, valid enough for the coder validator to read its bucket."""
    fields: dict[str, Any] = {
        "payload": dict(BRIEF),
        "provenance": AUTHORED,
        "tags": (f"bucket:{bucket.value}",),
        "difficulty": dict(EASY_AXES),
        "split": None,
    }
    fields.update(overrides)
    return Case(case_id=case_id, site_id=CODER_SITE_ID, **fields)


def _committed() -> tuple[Case, ...]:
    """Every case the repo ships for the coder site, loaded through the real loader."""
    return CoderCaseProvider(cases_root=CASES_ROOT).load(CODER_SITE_ID)


def _canaries() -> tuple[Case, ...]:
    """The committed canary bucket."""
    return tuple(case for case in _committed() if bucket_of(case) is Bucket.CANARY)


def _tracked(path: str) -> list[str]:
    """Every path git tracks under ``path``."""
    listed = subprocess.run(
        ["git", "ls-files", path], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return listed.stdout.split()


def _ignored(path: str) -> bool:
    """Whether git would ignore ``path`` — the deny-by-default allowlist, asserted at source."""
    return subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO_ROOT).returncode == 0


# ── the committed scaffold ────────────────────────────────────────────────────────────────


def test_the_coder_scaffold_ships_a_readme_beside_the_edge_and_canary_bucket_directories():
    assert CODER_README.is_file()
    assert all((CODER_ROOT / bucket.value).is_dir() for bucket in COMMITTED_BUCKETS)


def test_the_committed_buckets_are_the_curated_ones_and_the_local_buckets_the_harvested_ones():
    assert COMMITTED_BUCKETS == (Bucket.EDGE, Bucket.CANARY)
    assert LOCAL_BUCKETS == (Bucket.FIELD, Bucket.REPLAY)


def test_a_field_case_harvested_from_an_operators_run_is_gitignored():
    assert _ignored(f"cases/{CODER_SITE_ID}/field/mined-case.yaml")


def test_a_replay_case_of_an_operators_own_rejection_is_gitignored():
    assert _ignored(f"cases/{CODER_SITE_ID}/replay/mined-case.yaml")


def test_a_case_dropped_beside_the_buckets_rather_than_into_one_is_gitignored():
    """Deny-by-default survives the allowlist: only the two curated buckets are re-included."""
    assert _ignored(f"cases/{CODER_SITE_ID}/loose-case.yaml")


def test_the_reviewed_canary_bucket_is_tracked_in_git():
    tracked = _tracked(f"cases/{CODER_SITE_ID}/canary")

    assert tracked
    assert all(path.endswith(".yaml") for path in tracked)


def test_the_coder_readme_and_the_edge_bucket_are_tracked_in_git():
    tracked = _tracked(f"cases/{CODER_SITE_ID}")

    assert f"cases/{CODER_SITE_ID}/README.md" in tracked
    assert any(path.startswith(f"cases/{CODER_SITE_ID}/edge/") for path in tracked)


def test_no_tracked_case_path_sits_outside_the_readmes_and_the_two_committed_buckets():
    allowed = ("cases/README.md", f"cases/{CODER_SITE_ID}/README.md")
    buckets = tuple(f"cases/{CODER_SITE_ID}/{bucket.value}/" for bucket in COMMITTED_BUCKETS)
    tracked = _tracked("cases")

    assert all(path in allowed or path.startswith(buckets) for path in tracked), tracked


# ── what the coder README documents ───────────────────────────────────────────────────────


def test_the_coder_readme_documents_every_brief_field_and_the_optional_fixed_oracle():
    text = CODER_README.read_text(encoding="utf-8")

    assert all(key in text for key in REQUIRED_BRIEF_KEYS)
    assert "scenario_spec" in text


def test_the_coder_readme_documents_every_bucket_by_name():
    text = CODER_README.read_text(encoding="utf-8")

    assert all(f"bucket:{bucket.value}" in text or f"`{bucket.value}`" in text for bucket in Bucket)


def test_the_coder_readme_documents_every_axis_with_its_whole_closed_value_set():
    text = CODER_README.read_text(encoding="utf-8")

    for axis, levels in AXIS_LEVELS.items():
        assert axis.value in text, axis
        assert all(level in text for level in levels), axis


def test_the_coder_readme_documents_both_provenance_forms():
    text = CODER_README.read_text(encoding="utf-8")

    assert "mined:" in text and "authored:" in text


def test_the_coder_readme_documents_the_stamped_split_rule_and_the_bucket_directory_layout():
    text = CODER_README.read_text(encoding="utf-8")

    assert "split" in text and "stamp" in text.lower()
    assert f"cases/{CODER_SITE_ID}/<bucket>/" in text


# ── the loader over the bucket sub-directories ────────────────────────────────────────────


def test_every_committed_coder_case_loads_clean_through_the_provider_and_the_coder_validator():
    """A malformed committed case fails the suite here rather than skewing a benchmark later."""
    cases = _committed()

    assert cases
    assert all(isinstance(coder_payload(case).brief, StrategyBrief) for case in cases)


def test_the_loader_walks_the_bucket_directories_the_flat_provider_would_never_look_in(tmp_path):
    _write(tmp_path, Bucket.CANARY, "plain-crossover")
    _write(tmp_path, Bucket.EDGE, "higher-timeframe")

    loaded = CoderCaseProvider(cases_root=tmp_path).load(CODER_SITE_ID)

    assert [case.case_id for case in loaded] == ["higher-timeframe", "plain-crossover"]


def test_a_loaded_case_carries_the_bucket_of_the_directory_it_was_filed_in(tmp_path):
    _write(tmp_path, Bucket.EDGE, "higher-timeframe")

    (case,) = CoderCaseProvider(cases_root=tmp_path).load(CODER_SITE_ID)

    assert bucket_of(case) is Bucket.EDGE


def test_a_case_whose_bucket_tag_disagrees_with_its_directory_is_refused_naming_both(tmp_path):
    path = _write(tmp_path, Bucket.CANARY, "mislabelled", tags=["bucket:field"])

    with pytest.raises(MalformedCase) as refusal:
        CoderCaseProvider(cases_root=tmp_path).load(CODER_SITE_ID)

    assert str(path) in str(refusal.value)
    assert "field" in str(refusal.value) and "canary" in str(refusal.value)


def test_a_case_sitting_beside_the_buckets_rather_than_in_one_is_refused_not_left_unread(tmp_path):
    root = tmp_path / CODER_SITE_ID
    root.mkdir(parents=True)
    stray = root / "loose-case.yaml"
    stray.write_text(yaml.safe_dump(_document(Bucket.CANARY)), "utf-8")

    with pytest.raises(MalformedCase) as refusal:
        CoderCaseProvider(cases_root=tmp_path).load(CODER_SITE_ID)

    assert str(stray) in str(refusal.value)


def test_a_case_saved_with_the_wrong_extension_is_refused_rather_than_left_unread(tmp_path):
    _write(tmp_path, Bucket.CANARY, "plain-crossover")
    stray = tmp_path / CODER_SITE_ID / "canary" / "typo.yml"
    stray.write_text(yaml.safe_dump(_document(Bucket.CANARY)), "utf-8")

    with pytest.raises(MalformedCase) as refusal:
        CoderCaseProvider(cases_root=tmp_path).load(CODER_SITE_ID)

    assert str(stray) in str(refusal.value)


def test_a_directory_that_is_not_a_declared_bucket_is_refused_naming_it(tmp_path):
    _write(tmp_path, Bucket.CANARY, "plain-crossover")
    (tmp_path / CODER_SITE_ID / "drafts").mkdir()

    with pytest.raises(MalformedCase) as refusal:
        CoderCaseProvider(cases_root=tmp_path).load(CODER_SITE_ID)

    assert "drafts" in str(refusal.value)


def test_a_bucket_directory_that_does_not_exist_yet_contributes_no_cases(tmp_path):
    _write(tmp_path, Bucket.CANARY, "plain-crossover")

    loaded = CoderCaseProvider(cases_root=tmp_path).load(CODER_SITE_ID)

    assert [case.case_id for case in loaded] == ["plain-crossover"]


def test_an_absent_coder_corpus_is_refused_rather_than_read_as_an_empty_one(tmp_path):
    with pytest.raises(MissingCorpus) as refusal:
        CoderCaseProvider(cases_root=tmp_path).load(CODER_SITE_ID)

    assert str(tmp_path / CODER_SITE_ID) in str(refusal.value)


def test_the_coder_loader_refuses_a_site_it_does_not_read(tmp_path):
    _write(tmp_path, Bucket.CANARY, "plain-crossover")

    with pytest.raises(ValueError, match="distill"):
        CoderCaseProvider(cases_root=tmp_path).load("distill")


def test_the_committed_corpus_loads_as_one_corpus_with_a_content_digest(tmp_path):
    corpus = load_coder_corpus(CASES_ROOT)

    assert corpus.site_id == CODER_SITE_ID
    assert corpus.identity.case_count == len(corpus)
    assert corpus.digest == load_coder_corpus(CASES_ROOT).digest


# ── the split: stamped once, read forever ─────────────────────────────────────────────────


def test_every_committed_case_file_carries_a_stamped_split_in_the_file_itself():
    files = sorted(CODER_ROOT.rglob("*.yaml"))

    assert files
    for path in files:
        assert yaml.safe_load(path.read_text(encoding="utf-8")).get("split") in {
            "tuning",
            "holdout",
        }, path


def test_loading_the_committed_corpus_never_re_deals_a_stamped_split():
    stamped = {
        path.stem: yaml.safe_load(path.read_text(encoding="utf-8"))["split"]
        for path in sorted(CODER_ROOT.rglob("*.yaml"))
    }

    corpus = load_coder_corpus(CASES_ROOT)

    assert {case.case_id: case.split.value for case in corpus.cases if case.split} == stamped


def test_an_unassigned_newcomer_cannot_move_a_committed_case_off_its_stamped_split():
    """The property the stamp buys: growing the corpus re-deals nothing that already shipped."""
    committed = _committed()
    before = {case.case_id: case.split for case in Corpus(CODER_SITE_ID, committed).cases}

    grown = Corpus(CODER_SITE_ID, (*committed, _case("newcomer-case")))

    assert {case.case_id: case.split for case in grown.cases if case.case_id in before} == before


def test_the_stamped_splits_are_exactly_what_the_authoring_rule_deals_from_unassigned_cases():
    unassigned = tuple(
        Case(
            case_id=case.case_id,
            site_id=case.site_id,
            payload=dict(case.payload),
            provenance=case.provenance,
            tags=case.tags,
            difficulty=dict(case.difficulty),
        )
        for case in _committed()
    )

    assert deal_splits(unassigned) == {case.case_id: case.split for case in _committed()}


def test_the_deal_is_the_same_whatever_order_the_cases_arrive_in():
    cases = tuple(_case(f"case-{index:02d}") for index in range(10))

    assert deal_splits(cases) == deal_splits(tuple(reversed(cases)))


def test_the_bucket_is_part_of_the_stratum_key_so_two_buckets_are_dealt_apart():
    """Same axes, different buckets: one stratum of two would deal a holdout, two of one do not."""
    canary = _case("plain-crossover", Bucket.CANARY)
    edge = _case("higher-timeframe", Bucket.EDGE)

    assert deal_splits((canary, edge)) == {
        "plain-crossover": Split.TUNING,
        "higher-timeframe": Split.TUNING,
    }


def test_a_bucket_is_dealt_at_the_holdout_share_within_one_case():
    canaries = tuple(_case(f"canary-{index:02d}") for index in range(10))

    dealt = deal_splits(canaries)
    holdout = sum(1 for split in dealt.values() if split is Split.HOLDOUT)

    assert abs(holdout - len(canaries) * HOLDOUT_SHARE) <= 1


def test_every_committed_bucket_sits_within_one_case_of_the_holdout_share():
    for bucket in COMMITTED_BUCKETS:
        cases = [case for case in _committed() if bucket_of(case) is bucket]
        holdout = sum(1 for case in cases if case.split is Split.HOLDOUT)

        assert abs(holdout - len(cases) * HOLDOUT_SHARE) <= 1, bucket


def test_the_stamping_tool_writes_the_dealt_split_into_a_file_that_declares_none(tmp_path):
    path = _write(tmp_path, Bucket.CANARY, "plain-crossover")

    stamped = stamp_splits(tmp_path)

    assert stamped == (path,)
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["split"] in {"tuning", "holdout"}


def test_the_stamping_tool_leaves_an_already_stamped_file_byte_for_byte_alone(tmp_path):
    path = _write(tmp_path, Bucket.CANARY, "plain-crossover", split="holdout")
    before = path.read_bytes()

    assert stamp_splits(tmp_path) == ()
    assert path.read_bytes() == before


def test_the_stamping_tool_is_idempotent_so_re_running_it_re_deals_nothing(tmp_path):
    for index in range(6):
        _write(tmp_path, Bucket.CANARY, f"canary-{index:02d}")
    stamp_splits(tmp_path)
    after = {path: path.read_bytes() for path in sorted((tmp_path / CODER_SITE_ID).rglob("*.yaml"))}

    assert stamp_splits(tmp_path) == ()
    assert {path: path.read_bytes() for path in after} == after


# ── the canary bucket ─────────────────────────────────────────────────────────────────────


def test_the_canary_bucket_ships_six_cases():
    assert len(_canaries()) == 6


def test_every_canary_records_the_committed_seed_it_adapts_and_that_seed_ships():
    canaries = _canaries()

    assert all(ADAPTS_TAG in case.tags for case in canaries)
    assert (REPO_ROOT / "strategies" / f"{SEED}.py").is_file()


def test_every_canary_declares_a_distinct_single_rule_change():
    rules = [tag for case in _canaries() for tag in case.tags if tag.startswith(RULE_TAG_PREFIX)]

    assert len(rules) == len(_canaries())
    assert len(set(rules)) == len(rules)


def test_every_canary_is_labelled_on_all_seven_axes():
    for case in _canaries():
        payload = coder_payload(case)

        assert {axis.value for axis in Axis} == set(payload.difficulty), case.case_id


def test_no_canary_sits_at_the_hardest_level_of_any_axis():
    """A canary a model can fail on merit is not one — it is an edge case in the wrong bucket."""
    for case in _canaries():
        payload = coder_payload(case)

        for axis in Axis:
            levels = AXIS_LEVELS[axis]
            assert levels.index(payload.level(axis)) < len(levels) - 1, (case.case_id, axis)


def test_every_canary_brief_is_field_for_field_valid_strategy_brief_material():
    for case in _canaries():
        brief = coder_payload(case).brief

        assert isinstance(brief, StrategyBrief)
        assert all(getattr(brief, key).strip() for key in REQUIRED_BRIEF_KEYS), case.case_id


def test_no_canary_ships_a_fixed_oracle_so_the_coder_owns_its_own_tapes():
    for case in _canaries():
        payload = coder_payload(case)

        assert payload.scenario_spec is None, case.case_id
        assert payload.level(Axis.ORACLE_MODE) == "authored", case.case_id


def test_every_canary_was_authored_rather_than_mined_from_an_operators_run():
    for case in _canaries():
        assert case.provenance.kind is ProvenanceKind.AUTHORED, case.case_id
        assert case.provenance.authored_on is not None
