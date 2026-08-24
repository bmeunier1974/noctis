"""The strategy file's research record as text: ``StrategyHeader``, its parse (#312) and the
``stamp_header`` write side (#314).

String in, string (or value) out — no ``tmp_path``, no gate, no I/O in the module under test. It
is pure by rule (``noctis.strategies.header`` imports the standard library and nothing else), and
the last two tests here are what pins that rule, the way ``tests/test_eval_boundary.py`` pins the
eval one. The characterization section reads the committed seeds and their goldens as *text
inputs* to that same pure function: the goldens were rendered by the private
``library._render_header_fields`` at pre-epic commit ``1ab6da8``, so a byte of drift there means
the stamp was re-typeset, not moved.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from noctis.strategies.header import (
    HEADER_FIELDS,
    VALID_STATUSES,
    HeaderError,
    StrategyHeader,
    parse_header,
    stamp_header,
)

HEADER_SOURCE = Path(__file__).resolve().parents[1] / "src" / "noctis" / "strategies" / "header.py"


def source(docstring: str, body: str = "x = 1\n") -> str:
    """A strategy file's shape: a module docstring header, then code."""
    return f'"""{dedent(docstring).strip()}\n"""\n{body}'


# ── 1. The value: frozen, and legal by construction ──────────────────────────────────────


def test_the_default_header_is_a_legal_draft() -> None:
    header = StrategyHeader()

    assert header.status == "draft"
    assert header.thesis == ""
    assert header.style == ""
    assert header.symbols == []
    assert header.tuned is None


def test_a_header_cannot_be_mutated_after_construction() -> None:
    header = StrategyHeader(status="candidate")

    with pytest.raises(dataclasses.FrozenInstanceError):
        header.status = "champion"  # type: ignore[misc]


def test_a_status_outside_the_valid_set_cannot_be_constructed() -> None:
    with pytest.raises(HeaderError) as excinfo:
        StrategyHeader(status="shipped")

    assert str(excinfo.value) == (
        "header status 'shipped' invalid; want one of "
        "('draft', 'candidate', 'champion', 'rejected')"
    )


def test_the_header_error_is_a_value_error() -> None:
    """Callers that only knew ``ValueError`` keep catching what they always caught."""
    assert issubclass(HeaderError, ValueError)


def test_every_valid_status_constructs() -> None:
    assert VALID_STATUSES == ("draft", "candidate", "champion", "rejected")
    assert [StrategyHeader(status=s).status for s in VALID_STATUSES] == list(VALID_STATUSES)


def test_the_header_fields_are_the_vocabulary_the_file_declares() -> None:
    assert HEADER_FIELDS == ("status", "style", "symbols", "tuned")


def test_to_dict_carries_every_field_and_copies_the_symbol_list() -> None:
    header = StrategyHeader(
        thesis="Buy dips.", status="champion", style="mean-reversion", symbols=["AAPL"], tuned="d"
    )

    rendered = header.to_dict()

    assert rendered == {
        "thesis": "Buy dips.",
        "status": "champion",
        "style": "mean-reversion",
        "symbols": ["AAPL"],
        "tuned": "d",
    }
    assert rendered["symbols"] is not header.symbols


# ── 2. The read side: parse, tolerant of the file and strict on the value ─────────────────


def test_parse_reads_the_thesis_and_the_declared_fields() -> None:
    header = StrategyHeader.parse(
        source(
            """
            Toy probe: long above its own moving average.

            status: candidate
            style: momentum
            symbols: AAPL MSFT
            tuned: 2026-07-04
            """
        )
    )

    assert header == StrategyHeader(
        thesis="Toy probe: long above its own moving average.",
        status="candidate",
        style="momentum",
        symbols=["AAPL", "MSFT"],
        tuned="2026-07-04",
    )


def test_the_thesis_is_the_first_paragraph_joined_into_one_line() -> None:
    header = StrategyHeader.parse(
        source(
            """
            Shallow dips inside an uptrend snap back,
            because forced sellers are done before the trend is.

            An elaboration paragraph is not the thesis.

            status: draft
            """
        )
    )

    assert header.thesis == (
        "Shallow dips inside an uptrend snap back, "
        "because forced sellers are done before the trend is."
    )


def test_a_trailing_comment_is_stripped_from_a_field_value() -> None:
    header = StrategyHeader.parse(
        source(
            """
            Toy probe.

            status: candidate            # draft | candidate | champion | rejected
            style: mean-reversion        # momentum | mean-reversion | breakout | ...
            """
        )
    )

    assert header.status == "candidate"
    assert header.style == "mean-reversion"


def test_symbols_split_on_commas_or_whitespace_and_upper_case() -> None:
    header = StrategyHeader.parse(
        source(
            """
            Toy probe.

            symbols: aapl, msft  nvda,tsla     # the panel it was tuned on
            """
        )
    )

    assert header.symbols == ["AAPL", "MSFT", "NVDA", "TSLA"]


def test_an_empty_tuned_field_reads_as_none() -> None:
    header = StrategyHeader.parse(
        source(
            """
            Toy probe.

            status: draft
            tuned:
            """
        )
    )

    assert header.tuned is None


def test_a_file_with_no_docstring_reads_as_the_default_header() -> None:
    assert StrategyHeader.parse("x = 1\n") == StrategyHeader()


def test_a_file_that_does_not_parse_reads_as_the_default_header() -> None:
    """Tolerant of the file: a broken draft is still listed, not an exception at the reader."""
    assert StrategyHeader.parse("def (:\n") == StrategyHeader()


def test_a_declared_status_outside_the_valid_set_raises() -> None:
    """Strict on the value: a missing status defaults, a *wrong* one is a different case."""
    with pytest.raises(HeaderError, match="header status 'shipped' invalid"):
        StrategyHeader.parse(
            source(
                """
                Toy probe.

                status: shipped
                """
            )
        )


def test_a_header_that_declares_no_status_defaults_to_draft() -> None:
    header = StrategyHeader.parse(
        source(
            """
            Toy probe.

            style: momentum
            """
        )
    )

    assert header.status == "draft"
    assert header.style == "momentum"


def test_an_unknown_field_line_is_not_a_header_field() -> None:
    header = StrategyHeader.parse(
        source(
            """
            Toy probe.

            statuses: shipped
            """
        )
    )

    assert header == StrategyHeader(thesis="Toy probe.")


# ── 3. The write side: the stamp ─────────────────────────────────────────────────────────


def test_the_stamp_replaces_a_declared_field_in_place() -> None:
    before = (
        '"""Toy probe.\n'
        "\n"
        "    status: draft                # draft | candidate | champion | rejected\n"
        "    style: momentum              # the family it belongs to\n"
        '"""\n'
        "x = 1\n"
    )

    after = stamp_header(before, status="champion")

    assert after == (
        '"""Toy probe.\n'
        "\n"
        "    status: champion\n"
        "    style: momentum              # the family it belongs to\n"
        '"""\n'
        "x = 1\n"
    )


def test_a_field_left_none_is_not_touched_and_a_stamp_with_nothing_to_write_is_a_no_op() -> None:
    before = source(
        """
        Toy probe.

        status: candidate            # draft | candidate | champion | rejected
        style: mean-reversion
        symbols: AAPL MSFT
        tuned: 2026-07-04
        """
    )

    assert stamp_header(before) == before
    assert stamp_header(before, tuned="2026-01-01") == before.replace(
        "tuned: 2026-07-04", "tuned: 2026-01-01"
    )


def test_missing_fields_are_inserted_in_header_field_order_before_the_closing_quotes() -> None:
    before = source("Toy probe.")

    after = stamp_header(before, tuned="2026-01-01", symbols=["AAPL", "MSFT"], status="champion")

    assert after == source(
        """
        Toy probe.

        status: champion
        symbols: AAPL MSFT
        tuned: 2026-01-01
        """
    )


def test_an_inserted_field_packs_against_the_fields_already_declared() -> None:
    before = source(
        """
        Toy probe.

        status: candidate
        style: momentum
        """
    )

    after = stamp_header(before, tuned="2026-01-01")

    assert after == source(
        """
        Toy probe.

        status: candidate
        style: momentum
        tuned: 2026-01-01
        """
    )


def test_a_one_line_docstring_is_split_open_to_hold_the_fields() -> None:
    after = stamp_header('"""Toy probe."""\nx = 1\n', status="champion")

    assert after == source(
        """
        Toy probe.

        status: champion
        """
    )


def test_symbols_render_space_joined_and_a_ready_made_string_passes_through() -> None:
    before = source("Toy probe.")

    assert "symbols: AAPL MSFT NVDA\n" in stamp_header(before, symbols=["AAPL", "MSFT", "NVDA"])
    assert "symbols: AAPL MSFT NVDA\n" in stamp_header(before, symbols=("AAPL", "MSFT", "NVDA"))
    assert "symbols: AAPL MSFT NVDA\n" in stamp_header(before, symbols="AAPL MSFT NVDA")


def test_an_illegal_status_is_refused_before_any_edit() -> None:
    before = source(
        """
        Toy probe.

        status: draft
        """
    )

    with pytest.raises(HeaderError) as excinfo:
        stamp_header(before, status="shipped", tuned="2026-01-01")

    assert str(excinfo.value) == (
        "header status 'shipped' invalid; want one of "
        "('draft', 'candidate', 'champion', 'rejected')"
    )
    assert before == '"""Toy probe.\n\nstatus: draft\n"""\nx = 1\n'  # nothing was written


def test_the_status_is_checked_before_the_file_is_even_looked_at() -> None:
    """Refusal order, pinned: a bad status loses to nothing — not even a missing docstring."""
    with pytest.raises(HeaderError, match="header status 'shipped' invalid"):
        stamp_header("x = 1\n", status="shipped")


def test_a_source_with_no_module_docstring_cannot_be_stamped() -> None:
    with pytest.raises(HeaderError) as excinfo:
        stamp_header("x = 1\n", status="champion")

    assert str(excinfo.value) == "strategy file has no module docstring header"


def test_a_one_line_docstring_with_unusual_quoting_is_refused() -> None:
    with pytest.raises(HeaderError) as excinfo:
        stamp_header('r"""Toy probe."""\nx = 1\n', status="champion")

    assert str(excinfo.value) == "cannot rewrite docstring header (unusual quoting)"


def test_an_unknown_field_is_a_type_error_not_a_silent_drop() -> None:
    with pytest.raises(TypeError):
        stamp_header(source("Toy probe."), statuses="champion")  # type: ignore[call-arg]


def test_what_the_stamp_writes_is_what_the_parse_reads_back() -> None:
    stamped = stamp_header(
        source("Toy probe."), status="rejected", style="momentum", symbols=["AAPL"], tuned="d"
    )

    assert parse_header(stamped) == StrategyHeader(
        thesis="Toy probe.", status="rejected", style="momentum", symbols=["AAPL"], tuned="d"
    )


# ── 4. The move, pinned: the committed seeds stamp exactly as they always did ─────────────

SEEDS = sorted((Path(__file__).resolve().parents[1] / "strategies").glob("*.py"))
STAMP_GOLDENS = Path(__file__).resolve().parent / "fixtures" / "header_stamp"


@pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: seed.stem)
def test_a_committed_seed_stamps_byte_for_byte_as_the_library_renderer_did(seed: Path) -> None:
    """Characterization: the goldens were rendered by ``library._render_header_fields`` at
    ``1ab6da8`` (pre-epic). #314 moved that code here; a byte of difference is a rewrite."""
    stamped = stamp_header(
        seed.read_text(encoding="utf-8"),
        status="champion",
        symbols=["AAPL", "MSFT"],
        tuned="2026-01-01",
    )

    golden = STAMP_GOLDENS / f"{seed.stem}.stamped.py.txt"
    assert stamped == golden.read_text(encoding="utf-8")


def test_the_characterization_covers_every_committed_seed() -> None:
    assert [seed.stem for seed in SEEDS] == sorted(
        path.name.removesuffix(".stamped.py.txt") for path in STAMP_GOLDENS.glob("*.stamped.py.txt")
    )


# ── 5. The names the library still exports ───────────────────────────────────────────────


def test_parse_header_is_the_bound_alias_of_the_one_parser() -> None:
    assert parse_header("x = 1\n") == StrategyHeader()
    assert parse_header.__doc__ == StrategyHeader.parse.__doc__


def test_the_library_re_exports_the_header_names_it_always_exported() -> None:
    from noctis.strategies import library

    assert library.StrategyHeader is StrategyHeader
    assert library.parse_header is parse_header
    assert library.HEADER_FIELDS is HEADER_FIELDS
    assert library.VALID_STATUSES is VALID_STATUSES


# ── 6. The purity rule, pinned ───────────────────────────────────────────────────────────


def test_the_header_module_imports_only_the_standard_library() -> None:
    tree = ast.parse(HEADER_SOURCE.read_text(encoding="utf-8"), filename=str(HEADER_SOURCE))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])

    assert roots
    assert roots <= sys.stdlib_module_names


def test_the_header_modules_whole_import_closure_is_the_standard_library() -> None:
    """It loads as a bare file, off any package: nothing behind it to drag in."""
    probe = (
        "import importlib.util, json, sys\n"
        "baseline = set(sys.modules)\n"
        "spec = importlib.util.spec_from_file_location('header_probe', sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules['header_probe'] = module\n"
        "spec.loader.exec_module(module)\n"
        "pulled = {name.split('.')[0] for name in set(sys.modules) - baseline}\n"
        "print(json.dumps(sorted(pulled - sys.stdlib_module_names - {'header_probe'})))\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe, str(HEADER_SOURCE)],
        capture_output=True,
        text=True,
        check=True,
    )

    assert json.loads(completed.stdout) == []
