"""The failed-attempts reader (#216): a run's ``failed/`` folder read back and broken down.

Every record here comes from a file the **real** :class:`~noctis.research.failed_store.
FailedAttemptStore` wrote — never a hand-built header string — because a reader tested against its
own idea of the format is a reader that keeps passing after the writer rewords itself. The store
writes, the reader reads, and the assertions are external behaviour only: the records that come
back, the counts and shares of a breakdown, the lines a rendering carries, and the warnings a
folder of junk produces.
"""

from __future__ import annotations

import ast
from pathlib import Path

import noctis.eval.failed_attempts as failed_attempts_module
from noctis.eval.coder_taxonomy import OTHER, knob_for
from noctis.eval.failed_attempts import (
    ROT_THRESHOLD,
    MissingFailureFolder,
    failure_breakdown,
    read_failed_attempts,
    render_breakdown,
)
from noctis.research.failed_store import FailedAttemptStore

MODULE_SOURCE = Path(failed_attempts_module.__file__)

# Production gate wording, each landing in a different class of the coder vocabulary (#215).
IMPORT_ERROR = "ModuleNotFoundError: No module named 'talib'"
SCENARIO_ERROR = "StrategyValidationError: scenario 'rally': expected long at bar 40, got flat"
UNRECOGNISED = "the coder wandered off and wrote a haiku about the moon"


def _store(tmp_path: Path) -> FailedAttemptStore:
    """A store over a run-shaped working tier — the folder a real run's failures land in."""
    return FailedAttemptStore(tmp_path / "strategies" / "__tmp" / "failed")


def _records_from(store: FailedAttemptStore):
    return read_failed_attempts(store.root).records


def _counts(breakdown) -> dict[str, int]:
    return {tally.name: tally.count for tally in breakdown.tallies}


# ── round-trip against files the real store wrote ─────────────────────────────────────────
def test_an_attempt_the_store_recorded_reads_back_with_its_name_number_error_and_source(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.record("rsi_meanrev", 3, "class RsiMeanrev:\n    pass\n", IMPORT_ERROR)

    (record,) = _records_from(store)

    assert record.name == "rsi_meanrev"
    assert record.attempt == 3
    assert record.error == IMPORT_ERROR
    assert record.source == "class RsiMeanrev:\n    pass\n"


def test_a_multi_line_gate_error_reads_back_with_every_line_it_was_recorded_with(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    error = (
        "StrategyValidationError: scenario 'grind': expected flat, got long\n"
        "  observed: 4 entries between bars 12 and 60\n"
        "  the tape never leaves its band\n"
    ).rstrip("\n")
    store.record("band_probe", 1, "src", error)

    (record,) = _records_from(store)

    assert record.error == error
    assert record.error.splitlines()[1] == "  observed: 4 entries between bars 12 and 60"


def test_the_fixed_oracle_section_reads_back_when_the_store_recorded_one(tmp_path: Path) -> None:
    store = _store(tmp_path)
    oracle = "rally: trend(60) — enter long during leg 0\ngrind: flat(60) — never trade"
    store.record("spec_probe", 2, "src-body", SCENARIO_ERROR, oracle=oracle)

    (record,) = _records_from(store)

    assert record.oracle == oracle
    assert record.error == SCENARIO_ERROR
    assert record.source == "src-body"


def test_a_spec_less_attempt_reads_back_with_no_oracle_at_all(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record("hand_written", 1, "src", IMPORT_ERROR)

    (record,) = _records_from(store)

    assert record.oracle is None


def test_every_attempt_in_the_folder_is_read_back_exactly_once(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for attempt in range(1, 6):
        store.record("probe", attempt, f"source {attempt}", IMPORT_ERROR)

    records = _records_from(store)

    assert [record.attempt for record in records] == [1, 2, 3, 4, 5]
    assert [record.source for record in records] == [f"source {i}" for i in range(1, 6)]


def test_records_come_back_in_the_order_the_store_wrote_them(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record("alpha", 1, "a", IMPORT_ERROR)
    store.record("beta", 1, "b", IMPORT_ERROR)
    store.record("alpha", 2, "c", IMPORT_ERROR)

    records = _records_from(store)

    assert [(record.name, record.attempt) for record in records] == [
        ("alpha", 1),
        ("beta", 1),
        ("alpha", 2),
    ]


def test_a_record_names_the_file_it_was_read_from(tmp_path: Path) -> None:
    store = _store(tmp_path)
    path = store.record("rsi_meanrev", 7, "src", IMPORT_ERROR)

    (record,) = _records_from(store)

    assert record.filename == path.name


def test_a_source_carrying_the_stores_own_marker_lines_still_reads_back_whole(
    tmp_path: Path,
) -> None:
    # A strategy file may itself contain a comment that looks like the header's markers; the
    # header ends at the first source marker, so everything below it is source, verbatim.
    store = _store(tmp_path)
    source = "# gate error:\n# --- attempted source below ---\nclass X:\n    pass\n"
    store.record("mimic", 1, source, IMPORT_ERROR)

    (record,) = _records_from(store)

    assert record.source == source
    assert record.error == IMPORT_ERROR


# ── the breakdown: counts, shares, knobs, zero-hit classes ────────────────────────────────
def test_each_recorded_error_is_counted_into_its_taxonomy_class(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record("a", 1, "src", IMPORT_ERROR)
    store.record("b", 1, "src", IMPORT_ERROR)
    store.record("c", 1, "src", SCENARIO_ERROR)

    breakdown = failure_breakdown(_records_from(store))

    assert _counts(breakdown)["import_error"] == 2
    assert _counts(breakdown)["scenario_violation"] == 1
    assert breakdown.total == 3


def test_an_error_no_class_recognises_lands_in_the_escape_hatch(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record("a", 1, "src", UNRECOGNISED)

    breakdown = failure_breakdown(_records_from(store))

    assert _counts(breakdown)[OTHER] == 1


def test_every_declared_class_is_reported_even_when_nothing_matched_it(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record("a", 1, "src", IMPORT_ERROR)

    breakdown = failure_breakdown(_records_from(store))
    counts = _counts(breakdown)

    assert counts["truncated"] == 0
    assert counts["no_code_block"] == 0
    assert counts["warmup_too_large"] == 0
    assert len(breakdown.tallies) == len(counts) >= 10


def test_the_escape_hatch_is_the_last_class_reported() -> None:
    breakdown = failure_breakdown(())

    assert breakdown.tallies[-1].name == OTHER


def test_each_class_carries_the_knob_its_share_points_at() -> None:
    breakdown = failure_breakdown(())

    assert {tally.name: tally.knob for tally in breakdown.tallies} == {
        tally.name: knob_for(tally.name) for tally in breakdown.tallies
    }


def test_shares_are_fractions_of_the_attempt_total(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for _ in range(3):
        store.record("a", 1, "src", IMPORT_ERROR)
    store.record("b", 1, "src", SCENARIO_ERROR)

    breakdown = failure_breakdown(_records_from(store))
    shares = {tally.name: tally.share for tally in breakdown.tallies}

    assert shares["import_error"] == 0.75
    assert shares["scenario_violation"] == 0.25
    assert shares["truncated"] == 0.0
    assert sum(shares.values()) == 1.0


def test_a_breakdown_over_no_attempts_is_all_zeros_rather_than_a_division() -> None:
    breakdown = failure_breakdown(())

    assert breakdown.total == 0
    assert {tally.count for tally in breakdown.tallies} == {0}
    assert {tally.share for tally in breakdown.tallies} == {0.0}
    assert breakdown.escape_hatch_share == 0.0


def test_the_escape_hatch_share_is_readable_on_its_own(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for _ in range(3):
        store.record("a", 1, "src", IMPORT_ERROR)
    store.record("b", 1, "src", UNRECOGNISED)

    breakdown = failure_breakdown(_records_from(store))

    assert breakdown.escape_hatch_share == 0.25


# ── rendering: one screen, the escape hatch always on it ──────────────────────────────────
def test_the_rendering_names_every_class_with_its_count_share_and_knob(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record("a", 1, "src", IMPORT_ERROR)
    store.record("b", 1, "src", SCENARIO_ERROR)

    rendered = render_breakdown(failure_breakdown(_records_from(store)))

    for tally in failure_breakdown(_records_from(store)).tallies:
        assert tally.name in rendered
        assert tally.knob in rendered
    assert "50.0%" in rendered
    assert "2 attempts" in rendered


def test_a_zero_hit_class_is_still_visible_in_the_rendering(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record("a", 1, "src", IMPORT_ERROR)

    rendered = render_breakdown(failure_breakdown(_records_from(store)))
    line = next(row for row in rendered.splitlines() if row.startswith("truncated"))

    assert "0" in line
    assert "0.0%" in line


def test_the_escape_hatch_share_is_rendered_even_when_nothing_missed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record("a", 1, "src", IMPORT_ERROR)

    rendered = render_breakdown(failure_breakdown(_records_from(store)))

    assert "escape hatch" in rendered
    assert "0.0%" in rendered
    assert "ROT WARNING" not in rendered


def test_a_rot_warning_is_rendered_when_the_escape_hatch_share_exceeds_the_threshold(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    for _ in range(8):
        store.record("a", 1, "src", IMPORT_ERROR)
    for _ in range(2):
        store.record("b", 1, "src", UNRECOGNISED)

    breakdown = failure_breakdown(_records_from(store))
    rendered = render_breakdown(breakdown)

    assert breakdown.escape_hatch_share > ROT_THRESHOLD
    assert "ROT WARNING" in rendered
    assert f"{ROT_THRESHOLD:.0%}" in rendered


def test_no_rot_warning_when_the_escape_hatch_share_only_reaches_the_threshold(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    for _ in range(9):
        store.record("a", 1, "src", IMPORT_ERROR)
    store.record("b", 1, "src", UNRECOGNISED)

    breakdown = failure_breakdown(_records_from(store))

    assert breakdown.escape_hatch_share == ROT_THRESHOLD
    assert "ROT WARNING" not in render_breakdown(breakdown)
    assert "escape hatch" in render_breakdown(breakdown)


def test_a_rendering_over_no_attempts_says_so_instead_of_printing_a_verdict() -> None:
    rendered = render_breakdown(failure_breakdown(()))

    assert "0 attempts" in rendered
    assert "ROT WARNING" not in rendered
    assert "escape hatch" in rendered


def test_skipped_files_are_rendered_beside_the_counts_they_are_missing_from(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.record("a", 1, "src", IMPORT_ERROR)
    (store.root / "notes.txt").write_text("just a note\n", encoding="utf-8")

    result = read_failed_attempts(store.root)
    rendered = render_breakdown(failure_breakdown(result.records), skipped=result.warnings)

    assert "notes.txt" in rendered
    assert "skipped" in rendered.lower()


# ── robustness: junk in the folder, and folders that are not there ────────────────────────
def test_a_file_that_is_not_a_recorded_attempt_is_skipped_with_a_warning_naming_it(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.record("a", 1, "src", IMPORT_ERROR)
    (store.root / "000099-stray.py").write_text("print('not an attempt')\n", encoding="utf-8")

    result = read_failed_attempts(store.root)

    assert [record.name for record in result.records] == ["a"]
    assert [warning.filename for warning in result.warnings] == ["000099-stray.py"]
    assert "000099-stray.py" in result.warnings[0].line()


def test_a_truncated_attempt_file_is_skipped_rather_than_mis_binned(tmp_path: Path) -> None:
    # A header that never reaches the source marker (a killed write): counted nowhere, named.
    store = _store(tmp_path)
    path = store.record("a", 1, "src", IMPORT_ERROR)
    head = path.read_text(encoding="utf-8").split("# --- attempted source below ---")[0]
    (store.root / "000042-truncated-attempt1.py").write_text(head, encoding="utf-8")

    result = read_failed_attempts(store.root)

    assert len(result.records) == 1
    assert [warning.filename for warning in result.warnings] == ["000042-truncated-attempt1.py"]


def test_a_skipped_file_never_reaches_the_breakdown(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record("a", 1, "src", IMPORT_ERROR)
    (store.root / "junk.py").write_text("nothing like an attempt\n", encoding="utf-8")

    breakdown = failure_breakdown(read_failed_attempts(store.root).records)

    assert breakdown.total == 1


def test_a_warning_states_why_the_file_was_skipped(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record("a", 1, "src", IMPORT_ERROR)
    (store.root / "junk.py").write_text("nothing like an attempt\n", encoding="utf-8")

    (warning,) = read_failed_attempts(store.root).warnings

    assert warning.reason
    assert warning.reason in warning.line()


def test_a_subdirectory_of_the_failure_folder_is_skipped_with_a_warning(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record("a", 1, "src", IMPORT_ERROR)
    (store.root / "archive").mkdir()

    result = read_failed_attempts(store.root)

    assert len(result.records) == 1
    assert [warning.filename for warning in result.warnings] == ["archive"]


def test_an_empty_failure_folder_reports_zero_attempts_and_no_warnings(tmp_path: Path) -> None:
    folder = tmp_path / "strategies" / "__tmp" / "failed"
    folder.mkdir(parents=True)

    result = read_failed_attempts(folder)

    assert result.records == ()
    assert result.warnings == ()
    assert failure_breakdown(result.records).total == 0


def test_a_failure_folder_that_does_not_exist_refuses_naming_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "strategies" / "__tmp" / "failed"

    try:
        read_failed_attempts(missing)
    except MissingFailureFolder as refusal:
        assert str(missing) in str(refusal)
    else:  # pragma: no cover - the refusal is the behaviour under test
        raise AssertionError("a missing failure folder must refuse, not report an empty run")


def test_a_path_that_is_a_file_rather_than_a_folder_refuses_naming_it(tmp_path: Path) -> None:
    not_a_folder = tmp_path / "failed"
    not_a_folder.write_text("i am a file\n", encoding="utf-8")

    try:
        read_failed_attempts(not_a_folder)
    except MissingFailureFolder as refusal:
        assert str(not_a_folder) in str(refusal)
    else:  # pragma: no cover - the refusal is the behaviour under test
        raise AssertionError("a path that is not a folder must refuse")


# ── shape: no benchmark machinery, and folder I/O in exactly one place ────────────────────
def test_the_reader_reads_any_folder_path_with_no_benchmark_machinery_around_it(
    tmp_path: Path,
) -> None:
    # Not a run tree, not a corpus, not a harness — a bare folder a store happened to write into.
    store = FailedAttemptStore(tmp_path / "somewhere" / "else")
    store.record("probe", 1, "src", IMPORT_ERROR)

    result = read_failed_attempts(tmp_path / "somewhere" / "else")

    assert [record.name for record in result.records] == ["probe"]


def test_the_reader_imports_no_benchmark_machinery() -> None:
    tree = ast.parse(MODULE_SOURCE.read_text(encoding="utf-8"), filename=str(MODULE_SOURCE))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert imported & {"noctis.eval.coder_taxonomy"}
    assert not {name for name in imported if name.startswith("noctis.eval.")} - {
        "noctis.eval.coder_taxonomy"
    }


def test_folder_reading_is_confined_to_the_single_reader_function() -> None:
    # The pure core is the point: everything but one function is arithmetic over data, so a
    # breakdown can be re-derived from records without a filesystem in sight.
    filesystem = {
        "iterdir",
        "glob",
        "rglob",
        "read_text",
        "read_bytes",
        "write_text",
        "open",
        "is_dir",
        "is_file",
        "exists",
        "stat",
        "mkdir",
        "unlink",
    }
    tree = ast.parse(MODULE_SOURCE.read_text(encoding="utf-8"), filename=str(MODULE_SOURCE))
    reader = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "read_failed_attempts"
    )
    outside = [
        called
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for called in [
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else node.func.id
            if isinstance(node.func, ast.Name)
            else ""
        ]
        if called in filesystem
        and not (reader.lineno <= node.lineno <= (reader.end_lineno or reader.lineno))
    ]

    assert outside == []
