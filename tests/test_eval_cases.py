"""The case layer (#196): the frozen case, its strict validation, and the YAML provider.

A benchmark case is one small file a reviewer reads in a pull request, so the two things asserted
here are what that file may say and what it may never say. It carries the site it exercises, the
input payload, tags, difficulty axes, provenance and — once a corpus freezes one — a split; it
carries **no expected output**, because the site's own oracle is the expectation and a reference
solution in a corpus file rots the day production improves. Every refusal below is a defect that
would otherwise skew a benchmark number in silence.

The model and its validation are pure — the structural test at the bottom holds them to that, in
the technique the run-record suite uses on its pure builder — and the YAML filesystem provider is
the one thing here that touches a directory.
"""

from __future__ import annotations

import ast
import dataclasses
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from noctis.eval.case import (
    Case,
    CaseProvider,
    MalformedCase,
    Provenance,
    ProvenanceKind,
    Split,
    parse_case,
)
from noctis.eval.case_provider import CasePaths, MissingCorpus, YamlCaseProvider
from noctis.eval.registry import SITES, UnknownSite

CASE_SOURCE = Path(__file__).resolve().parents[1] / "src/noctis/eval/case.py"

# A run id of the shape the engine mints (``20260720T144233Z-a3f9c1``), so the mined provenance
# form is exercised against the real pattern rather than a lookalike.
RUN_ID = "20260720T144233Z-a3f9c1"


def _document(**overrides: Any) -> dict[str, Any]:
    """A complete, well-formed case document; keyword arguments replace one field each."""
    document: dict[str, Any] = {
        "site_id": "coder",
        "payload": {"brief": "a mean-reversion strategy", "symbols": ["AAPL", "MSFT"]},
        "provenance": "authored:2026-07-20",
        "tags": ["seed", "reversion"],
        "difficulty": {"reasoning": "hard", "novelty": "low"},
        "split": "tuning",
    }
    document.update(overrides)
    return document


def _write(directory: Path, name: str, document: Any) -> Path:
    """Write one case file, creating its site directory — the layout the provider reads."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    return path


def _corpus(root: Path, site_id: str = "coder") -> Path:
    """The per-site case directory under a cases root."""
    return root / site_id


# ── what a case carries ───────────────────────────────────────────────────────────────────


def test_a_case_file_round_trips_into_a_frozen_case_carrying_every_declared_field(tmp_path):
    _write(_corpus(tmp_path), "mean_reversion_brief", _document())

    (case,) = YamlCaseProvider(cases_root=tmp_path).load("coder")

    assert case == Case(
        case_id="mean_reversion_brief",
        site_id="coder",
        payload={"brief": "a mean-reversion strategy", "symbols": ("AAPL", "MSFT")},
        tags=("seed", "reversion"),
        difficulty={"reasoning": "hard", "novelty": "low"},
        provenance=Provenance(kind=ProvenanceKind.AUTHORED, reference="2026-07-20"),
        split=Split.TUNING,
    )


def test_a_case_id_is_stamped_from_the_file_name_so_the_directory_owns_case_identity(tmp_path):
    _write(_corpus(tmp_path), "gap_open_reversal", _document())

    (case,) = YamlCaseProvider(cases_root=tmp_path).load("coder")

    assert case.case_id == "gap_open_reversal"


def test_a_case_instance_refuses_field_assignment(tmp_path):
    _write(_corpus(tmp_path), "one", _document())
    (case,) = YamlCaseProvider(cases_root=tmp_path).load("coder")

    with pytest.raises(dataclasses.FrozenInstanceError):
        case.site_id = "decide"


def test_a_case_payload_is_read_only_including_its_nested_sequences(tmp_path):
    _write(_corpus(tmp_path), "one", _document(payload={"brief": "x", "symbols": ["AAPL"]}))
    (case,) = YamlCaseProvider(cases_root=tmp_path).load("coder")

    assert case.payload["symbols"] == ("AAPL",)
    with pytest.raises(TypeError):
        case.payload["brief"] = "y"


def test_tags_and_difficulty_axes_are_empty_when_a_file_declares_neither(tmp_path):
    document = _document()
    del document["tags"]
    del document["difficulty"]
    _write(_corpus(tmp_path), "bare", document)

    (case,) = YamlCaseProvider(cases_root=tmp_path).load("coder")

    assert (case.tags, dict(case.difficulty)) == ((), {})


# ── the expectation a case never carries ──────────────────────────────────────────────────


def test_the_case_type_declares_no_expected_output_field():
    """The oracle is the site's, not the corpus's — no field may hold a reference answer."""
    names = {field.name for field in dataclasses.fields(Case)}

    assert not [
        name
        for name in names
        for banned in ("expect", "gold", "answer", "solution", "reference_output")
        if banned in name
    ]


@pytest.mark.parametrize(
    "key", ["expected", "expected_output", "expectation", "gold", "answer", "solution"]
)
def test_a_file_carrying_an_expected_output_key_is_refused_by_that_key_name(tmp_path, key):
    _write(_corpus(tmp_path), "leaky", _document(**{key: "buy AAPL"}))

    with pytest.raises(MalformedCase) as refusal:
        YamlCaseProvider(cases_root=tmp_path).load("coder")

    assert key in str(refusal.value)
    assert "oracle" in str(refusal.value)


# ── refusals, loud and named ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("missing", ["site_id", "payload", "provenance"])
def test_a_missing_required_key_is_refused_naming_the_file_and_the_key(tmp_path, missing):
    document = _document()
    del document[missing]
    path = _write(_corpus(tmp_path), "incomplete", document)

    with pytest.raises(MalformedCase) as refusal:
        YamlCaseProvider(cases_root=tmp_path).load("coder")

    assert str(path) in str(refusal.value)
    assert missing in str(refusal.value)


def test_an_unknown_key_is_refused_naming_it_beside_the_keys_a_case_declares(tmp_path):
    _write(_corpus(tmp_path), "chatty", _document(notes="scratch thoughts"))

    with pytest.raises(MalformedCase) as refusal:
        YamlCaseProvider(cases_root=tmp_path).load("coder")

    assert "notes" in str(refusal.value)
    assert "difficulty" in str(refusal.value)


def test_every_defect_in_one_file_is_reported_in_a_single_refusal(tmp_path):
    """A bad case is fixed in one edit, never in a fix-one-rerun loop."""
    document = _document(tags="seed", provenance="hand-written", notes="scratch")
    del document["payload"]

    with pytest.raises(MalformedCase) as refusal:
        parse_case(document, case_id="messy", source="cases/coder/messy.yaml")

    message = str(refusal.value)
    assert all(defect in message for defect in ("payload", "tags", "provenance", "notes"))


@pytest.mark.parametrize("payload", [None, "a brief", ["a brief"], {}])
def test_a_payload_that_is_not_a_populated_mapping_is_refused(tmp_path, payload):
    with pytest.raises(MalformedCase) as refusal:
        parse_case(_document(payload=payload), case_id="one", source="one.yaml")

    assert "payload" in str(refusal.value)


def test_a_document_that_is_not_a_mapping_at_all_is_refused_naming_the_file(tmp_path):
    _write(_corpus(tmp_path), "listy", ["site_id: coder"])

    with pytest.raises(MalformedCase) as refusal:
        YamlCaseProvider(cases_root=tmp_path).load("coder")

    assert "listy.yaml" in str(refusal.value)


@pytest.mark.parametrize("tags", ["seed", ["seed", "seed"], ["seed", ""], [1]])
def test_tags_that_are_not_distinct_non_empty_strings_are_refused(tags):
    with pytest.raises(MalformedCase) as refusal:
        parse_case(_document(tags=tags), case_id="one", source="one.yaml")

    assert "tags" in str(refusal.value)


@pytest.mark.parametrize("difficulty", ["hard", {"reasoning": 3}, {"reasoning": None}, {3: "hard"}])
def test_difficulty_axes_that_are_not_a_name_to_level_mapping_of_strings_are_refused(difficulty):
    with pytest.raises(MalformedCase) as refusal:
        parse_case(_document(difficulty=difficulty), case_id="one", source="one.yaml")

    assert "difficulty" in str(refusal.value)


def test_a_case_naming_a_site_nothing_declares_is_refused_beside_the_declared_ids():
    with pytest.raises(MalformedCase) as refusal:
        parse_case(
            _document(site_id="oracle"),
            case_id="one",
            source="one.yaml",
            declared_site_ids=frozenset(SITES),
        )

    assert "oracle" in str(refusal.value)
    assert "coder" in str(refusal.value)


def test_a_site_id_is_taken_as_written_when_no_declared_set_is_supplied():
    """The pure validator cross-checks only what it is given; the provider supplies the registry."""
    case = parse_case(_document(site_id="scratch"), case_id="one", source="one.yaml")

    assert case.site_id == "scratch"


# ── provenance ────────────────────────────────────────────────────────────────────────────


def test_the_mined_provenance_form_parses_into_its_run_id():
    provenance = Provenance.parse(f"mined:{RUN_ID}")

    assert (provenance.kind, provenance.reference) == (ProvenanceKind.MINED, RUN_ID)


def test_the_authored_provenance_form_parses_into_its_iso_date():
    provenance = Provenance.parse("authored:2026-07-20")

    assert (provenance.kind, provenance.reference) == (ProvenanceKind.AUTHORED, "2026-07-20")
    assert provenance.authored_on == date(2026, 7, 20)


def test_a_provenance_round_trips_through_the_text_a_case_file_carries():
    text = f"mined:{RUN_ID}"

    assert str(Provenance.parse(text)) == text


@pytest.mark.parametrize(
    "text",
    [
        "hand-written",
        "mined",
        "mined:",
        "mined:not-a-run-id",
        "mined:20260720T144233Z-A3F9C1",
        "authored:",
        "authored:20260720",
        "authored:2026-13-01",
        "authored:yesterday",
        "MINED:20260720T144233Z-a3f9c1",
        "",
    ],
)
def test_a_provenance_outside_the_two_declared_forms_is_refused(text):
    with pytest.raises(MalformedCase) as refusal:
        Provenance.parse(text)

    assert "mined:" in str(refusal.value) and "authored:" in str(refusal.value)


def test_a_bad_provenance_in_a_file_is_refused_naming_the_file(tmp_path):
    path = _write(_corpus(tmp_path), "aged", _document(provenance="authored:yesterday"))

    with pytest.raises(MalformedCase) as refusal:
        YamlCaseProvider(cases_root=tmp_path).load("coder")

    assert str(path) in str(refusal.value)


# ── the split #197 freezes ────────────────────────────────────────────────────────────────


def test_a_file_that_declares_no_split_loads_unassigned_so_a_corpus_can_freeze_one(tmp_path):
    document = _document()
    del document["split"]
    _write(_corpus(tmp_path), "unassigned", document)

    (case,) = YamlCaseProvider(cases_root=tmp_path).load("coder")

    assert case.split is None


def test_assigning_a_split_yields_a_new_case_and_leaves_the_loaded_one_untouched():
    case = parse_case(
        {**_document(), "split": None}, case_id="one", source="one.yaml"
    )  # an unassigned case

    assigned = case.assigned_to(Split.HOLDOUT)

    assert (assigned.split, case.split) == (Split.HOLDOUT, None)
    assert assigned.case_id == case.case_id


def test_a_case_whose_split_is_already_frozen_refuses_reassignment():
    case = parse_case(_document(split="holdout"), case_id="one", source="one.yaml")

    with pytest.raises(ValueError) as refusal:
        case.assigned_to(Split.TUNING)

    assert "holdout" in str(refusal.value)


@pytest.mark.parametrize("split", ["tune", "validation", "", 1])
def test_a_split_outside_tuning_and_holdout_is_refused(split):
    with pytest.raises(MalformedCase) as refusal:
        parse_case(_document(split=split), case_id="one", source="one.yaml")

    assert "tuning" in str(refusal.value) and "holdout" in str(refusal.value)


# ── the YAML filesystem provider ──────────────────────────────────────────────────────────


def test_the_provider_loads_a_site_directory_in_file_name_order(tmp_path):
    for name in ("gamma", "alpha", "beta"):
        _write(_corpus(tmp_path), name, _document())

    cases = YamlCaseProvider(cases_root=tmp_path).load("coder")

    assert [case.case_id for case in cases] == ["alpha", "beta", "gamma"]


def test_the_provider_loads_only_the_requested_site_directory(tmp_path):
    _write(_corpus(tmp_path, "coder"), "authoring", _document())
    _write(_corpus(tmp_path, "decide"), "verdict", _document(site_id="decide"))

    cases = YamlCaseProvider(cases_root=tmp_path).load("decide")

    assert [(case.case_id, case.site_id) for case in cases] == [("verdict", "decide")]


def test_a_case_whose_site_id_disagrees_with_its_directory_is_refused_naming_both(tmp_path):
    _write(_corpus(tmp_path, "coder"), "misfiled", _document(site_id="decide"))

    with pytest.raises(MalformedCase) as refusal:
        YamlCaseProvider(cases_root=tmp_path).load("coder")

    assert "coder" in str(refusal.value) and "decide" in str(refusal.value)


def test_the_provider_refuses_a_site_id_nothing_declares(tmp_path):
    _write(_corpus(tmp_path, "oracle"), "one", _document(site_id="oracle"))

    with pytest.raises(UnknownSite):
        YamlCaseProvider(cases_root=tmp_path).load("oracle")


def test_a_missing_case_directory_is_refused_naming_the_path_it_looked_for(tmp_path):
    with pytest.raises(MissingCorpus) as refusal:
        YamlCaseProvider(cases_root=tmp_path).load("coder")

    assert str(tmp_path / "coder") in str(refusal.value)


def test_an_existing_but_empty_case_directory_loads_no_cases(tmp_path):
    _corpus(tmp_path).mkdir(parents=True)

    assert YamlCaseProvider(cases_root=tmp_path).load("coder") == ()


def test_a_file_that_is_not_valid_yaml_is_refused_naming_the_file(tmp_path):
    _corpus(tmp_path).mkdir(parents=True)
    (_corpus(tmp_path) / "broken.yaml").write_text("site_id: [coder\n", encoding="utf-8")

    with pytest.raises(MalformedCase) as refusal:
        YamlCaseProvider(cases_root=tmp_path).load("coder")

    assert "broken.yaml" in str(refusal.value)


def test_a_case_file_saved_with_a_yml_extension_is_refused_rather_than_skipped(tmp_path):
    _corpus(tmp_path).mkdir(parents=True)
    (_corpus(tmp_path) / "typo.yml").write_text(yaml.safe_dump(_document()), encoding="utf-8")

    with pytest.raises(MalformedCase) as refusal:
        YamlCaseProvider(cases_root=tmp_path).load("coder")

    assert "typo.yml" in str(refusal.value)


def test_the_provider_reads_the_registry_it_is_given_rather_than_the_shipped_one(tmp_path):
    """The lookup takes its registry as an argument, like every other eval-layer lookup."""
    _write(_corpus(tmp_path, "coder"), "one", _document())

    with pytest.raises(UnknownSite):
        YamlCaseProvider(cases_root=tmp_path, registry={}).load("coder")


# ── the corpus tiers ──────────────────────────────────────────────────────────────────────


def test_a_bare_path_still_means_the_single_root_layout_it_always_meant(tmp_path):
    """Every caller that predates the tiers keeps working, because a path coerces to one tier."""
    assert CasePaths.coerce(tmp_path) == CasePaths((tmp_path,))
    assert CasePaths.coerce(CasePaths((tmp_path,))) == CasePaths((tmp_path,))


def test_only_the_tiers_that_hold_the_site_are_listed_and_in_precedence_order(tmp_path):
    committed, workspace, empty = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    _write(_corpus(committed), "shipped", _document())
    _write(_corpus(workspace), "mined", _document())

    paths = CasePaths((committed, empty, workspace))

    assert paths.directories("coder") == (_corpus(committed), _corpus(workspace))


def test_both_tiers_load_together_ordered_by_case_id(tmp_path):
    committed, workspace = tmp_path / "committed", tmp_path / "workspace"
    _write(_corpus(committed), "shipped-canary", _document())
    _write(_corpus(workspace), "mined-from-a-run", _document())

    cases = YamlCaseProvider(cases_root=CasePaths((committed, workspace))).load("coder")

    assert [case.case_id for case in cases] == ["mined-from-a-run", "shipped-canary"]


def test_a_case_id_in_both_tiers_is_the_later_tiers(tmp_path):
    """The strategy library's rule: an operator shadows a shipped case, never the other way."""
    committed, workspace = tmp_path / "committed", tmp_path / "workspace"
    _write(_corpus(committed), "same-name", _document(tags=["shipped"]))
    _write(_corpus(workspace), "same-name", _document(tags=["mine"]))

    (case,) = YamlCaseProvider(cases_root=CasePaths((committed, workspace))).load("coder")

    assert case.tags == ("mine",)


def test_a_site_no_tier_holds_is_refused_naming_every_root_it_looked_in(tmp_path):
    committed, workspace = tmp_path / "committed", tmp_path / "workspace"

    with pytest.raises(MissingCorpus) as refusal:
        YamlCaseProvider(cases_root=CasePaths((committed, workspace))).load("coder")

    assert str(_corpus(committed)) in str(refusal.value)
    assert str(_corpus(workspace)) in str(refusal.value)


def test_one_tier_is_enough_so_a_workspace_that_has_never_been_mined_still_reads(tmp_path):
    """The regression the tiers exist for: a fresh install reaches the committed corpus."""
    committed, workspace = tmp_path / "committed", tmp_path / "workspace"
    _write(_corpus(committed), "shipped", _document())

    cases = YamlCaseProvider(cases_root=CasePaths((committed, workspace))).load("coder")

    assert [case.case_id for case in cases] == ["shipped"]


# ── the loading seam ──────────────────────────────────────────────────────────────────────


def test_the_yaml_filesystem_provider_satisfies_the_case_provider_protocol(tmp_path):
    assert isinstance(YamlCaseProvider(cases_root=tmp_path), CaseProvider)


def test_a_stand_in_provider_with_the_same_load_surface_satisfies_the_protocol():
    """The seam a mined or generated corpus becomes later, without touching its consumers."""

    class _InMemoryProvider:
        def load(self, site_id: str) -> tuple[Case, ...]:
            return (parse_case(_document(site_id=site_id), case_id="one", source="memory"),)

    provider: CaseProvider = _InMemoryProvider()

    assert isinstance(provider, CaseProvider)
    assert provider.load("coder")[0].site_id == "coder"


def test_an_object_without_a_load_surface_is_not_a_case_provider():
    assert not isinstance(object(), CaseProvider)


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


def test_the_case_model_and_its_validation_reach_no_file_and_no_clock():
    """The load-bearing decision of this slice: the provider is the only thing that reads."""
    assert _imports(CASE_SOURCE) <= {
        "__future__",
        "collections",
        "dataclasses",
        "datetime",
        "enum",
        "typing",
        "types",
        "noctis",
    }
    text = CASE_SOURCE.read_text()
    for forbidden in ("open(", "Path(", "os.", "random", "datetime.now", "utcnow", "today("):
        assert forbidden not in text, forbidden
