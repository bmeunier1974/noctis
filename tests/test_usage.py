"""One home for usage accounting, pinned here (story #346, epic #341).

A completion reports its spend as four neutral token fields, and four numbers travel a long way:
the conversation loop rolls them up per session, the episode runner keeps one split per episode,
the ledger persists them, the price table bills each at its own rate, and the run record publishes
them. Before this story the *list of the four* was written out five times and two byte-identical
accumulators folded it; a renamed field would have silently priced part of a session at zero.

So :mod:`noctis.research.usage` is the one declaration for the research package — the field list,
the accumulator, and the provider-neutral token estimate the loop and the briefing builders both
size prompts with. This file is three halves, in the shape of ``tests/test_store_writers.py``:

* **A second declaration cannot appear quietly.** A static scan over ``src/noctis`` classifies
  every literal that spells exactly the four field names, wherever it sits, and names the module
  it is in. It reads the shape too, because the two are not the same claim: a *list* of the four
  is a re-listing (there may be exactly one, and it is this module's), while a *mapping* keyed by
  the four carries a second fact per field — which is what pricing's field→rate table is. The
  scan reads the parsed source, so prose that lists the four fields — this paragraph, or a module
  docstring explaining them — is not a declaration.
* **The two mirrors are named, and pinned equal.** ``research/pricing.py`` keys its field→rate
  mapping by the same four, and ``reporting/run_record.py`` spells them as the record's own field
  names, because the record writer must stay light enough never to import the research package
  (the run-tree boundary exists to keep that heaviness out). Neither may drift: each is compared
  directly against the canonical list here, which is the import they deliberately do not have.
* **Every consumer meets the public surface.** One accumulator, one estimator, and
  ``research/briefings.py`` imports the estimator by its public name rather than reaching into
  the agent loop for a ``_``-prefixed one.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from noctis.reporting import run_record
from noctis.research import briefings, pricing, usage
from noctis.research.usage import (
    APPROX_CHARS_PER_TOKEN,
    USAGE_FIELDS,
    accumulate_usage,
    estimate_tokens,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "noctis"

#: The canonical four, spelled out once more *here* on purpose: a test that read the list off the
#: module it is testing would agree with any list that module ever held.
FOUR_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)

#: The canonical declaration, and the two mirrors that stay by design — a module that cannot
#: import the research package (the record) and one whose keys carry a second fact (the rates).
CANONICAL = "noctis.research.usage"
RECORD_MIRROR = "noctis.reporting.run_record"
RATE_MIRROR = "noctis.research.pricing"
MIRRORS = (RECORD_MIRROR, RATE_MIRROR)

#: How a declaration spells the four: a bare re-listing, or a mapping that keys a second fact by
#: them. Only the first kind is a duplicate of this module's list.
LIST = "list"
MAPPING = "mapping"


@dataclass(frozen=True)
class Declaration:
    """One literal that spells exactly the four usage fields: what it is called, and where."""

    module: str
    name: str
    shape: str
    path: str
    lineno: int

    def line(self) -> str:
        return f"{self.module}.{self.name} ({self.shape}, {self.path}:{self.lineno})"


def field_declarations(package_root: Path) -> tuple[Declaration, ...]:
    """Every assignment under ``package_root`` whose literal value names exactly the four fields.

    ``package_root`` is the top-level package directory (``src/noctis`` for this repo, or a
    fabricated one in a test); dotted module names are derived from the root's own name, so a
    miniature tree reads exactly like the real one. A tuple, list, set or dict counts — the two
    mirrors have different shapes and both are declarations of the same list.

    Sorted by module, then line: the same tree always yields the same tuple, so a failure message
    is stable across machines.
    """
    root = Path(package_root)
    found: list[Declaration] = []
    for module, path in _sources(root):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = _assigned_names(node)
            shape, spelled = _literal_strings(getattr(node, "value", None))
            if not names or spelled != frozenset(FOUR_FIELDS):
                continue
            found.extend(
                Declaration(
                    module=module,
                    name=name,
                    shape=shape,
                    path=path.relative_to(root.parent).as_posix(),
                    lineno=node.lineno,
                )
                for name in names
            )
    return tuple(sorted(found, key=lambda decl: (decl.module, decl.lineno, decl.name)))


def field_lists(package_root: Path) -> tuple[Declaration, ...]:
    """The declarations that are a bare re-listing of the four — the ones there may be one of."""
    return tuple(decl for decl in field_declarations(package_root) if decl.shape == LIST)


def function_definitions(package_root: Path, names: frozenset[str]) -> tuple[Declaration, ...]:
    """Every function under ``package_root`` defined under one of ``names`` — the twin-catcher."""
    root = Path(package_root)
    found = [
        Declaration(
            module=module,
            name=node.name,
            shape="def",
            path=path.relative_to(root.parent).as_posix(),
            lineno=node.lineno,
        )
        for module, path in _sources(root)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in names
    ]
    return tuple(sorted(found, key=lambda decl: (decl.module, decl.lineno)))


def imported_modules(path: Path) -> frozenset[str]:
    """The dotted module names one file imports, however the import is spelled."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return frozenset(found)


def imported_names(path: Path) -> frozenset[str]:
    """The names one file binds by importing them from somewhere else."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            found.update(alias.asname or alias.name for alias in node.names)
    return frozenset(found)


def report(declarations: tuple[Declaration, ...]) -> str:
    """The verdict a failing guard prints: every declaration found, then the rule it broke."""
    header = f"the usage field list is declared {len(declarations)} time(s):"
    rule = (
        f"  it is declared once, in {CANONICAL} — the two mirrors that cannot import it "
        f"({', '.join(MIRRORS)}) are pinned equal by test, not by a second list"
    )
    return "\n".join([header, *(f"  {decl.line()}" for decl in declarations), rule])


def _assigned_names(node: ast.AST) -> tuple[str, ...]:
    """The plain names a node assigns to — empty for anything that is not an assignment."""
    if isinstance(node, ast.Assign):
        return tuple(t.id for t in node.targets if isinstance(t, ast.Name))
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return (node.target.id,)
    return ()


def _literal_strings(value: ast.AST | None) -> tuple[str, frozenset[str]]:
    """How a literal collection spells its string constants: its shape, and the names in it.

    A dict contributes its keys (and is a ``mapping``); a tuple, list or set contributes its items
    (and is a ``list``). Anything else — a call, a name, a comprehension, a non-collection —
    contributes nothing, so only a list somebody actually typed out is ever judged a declaration.
    """
    if isinstance(value, ast.Dict):
        shape, elements = MAPPING, [key for key in value.keys if key is not None]
        typed = len(elements) == len(value.keys)
    elif isinstance(value, ast.Tuple | ast.List | ast.Set):
        shape, elements = LIST, list(value.elts)
        typed = True
    else:
        return "", frozenset()
    found = {
        el.value for el in elements if isinstance(el, ast.Constant) and isinstance(el.value, str)
    }
    return (shape, frozenset(found)) if typed and len(found) == len(elements) else ("", frozenset())


def _sources(root: Path) -> tuple[tuple[str, Path], ...]:
    """The package's modules as ``(dotted name, file)``, in a fixed order."""
    return tuple(sorted((_module_name(root, path), path) for path in root.rglob("*.py")))


def _module_name(root: Path, path: Path) -> str:
    parts = list(path.relative_to(root).parts)
    if parts[-1] == "__init__.py":
        parts.pop()
    else:
        parts[-1] = path.stem
    return ".".join([root.name, *parts])


def _tree(root: Path, modules: dict[str, str]) -> Path:
    """A miniature ``noctis`` package: a research subpackage, plus the modules a scenario names."""
    package = root / "noctis"
    (package / "research").mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text('"""a miniature engine."""\n')
    (package / "research" / "__init__.py").write_text('"""the research package."""\n')
    for rel, source in modules.items():
        target = package / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    return package


def _prose(text: str | None) -> str:
    """One docstring with its line wrapping collapsed, so reflowing a paragraph is no change."""
    return " ".join((text or "").split())


def _zeros() -> dict[str, int]:
    return dict.fromkeys(FOUR_FIELDS, 0)


# ── the canonical list ──────────────────────────────────────────────────────────────────────
def test_the_four_usage_fields_are_named_in_billing_order() -> None:
    assert USAGE_FIELDS == FOUR_FIELDS


def test_the_field_list_is_declared_exactly_once_in_the_research_package() -> None:
    """The story's point: one list, and every research consumer imports it — the agent's rollup,
    the episode's split and the ledger's persisted line included."""
    research = tuple(
        decl for decl in field_lists(SOURCE_ROOT) if decl.module.startswith("noctis.research")
    )

    assert [decl.module for decl in research] == [CANONICAL], report(research)


def test_the_only_other_re_listing_is_the_record_that_cannot_import_it() -> None:
    """A green above is a verdict only if the scan also found the mirror it tolerates: the record
    spells the four out because it must stay light enough never to import the research package."""
    listed = field_lists(SOURCE_ROOT)

    assert [decl.module for decl in listed] == [RECORD_MIRROR, CANONICAL], report(listed)


def test_the_only_mapping_keyed_by_the_four_is_the_price_table() -> None:
    """The other mirror, and the reason the scan reads shapes: pricing's keys are the field list,
    but its values are a second fact — which rate bills which field — that belongs there."""
    mapped = tuple(decl for decl in field_declarations(SOURCE_ROOT) if decl.shape == MAPPING)

    assert [decl.module for decl in mapped] == [RATE_MIRROR], report(mapped)


def test_the_price_table_bills_exactly_the_canonical_fields() -> None:
    """The mirror pinned by comparison, which is the import pricing deliberately does not need."""
    assert tuple(pricing.USAGE_FIELDS) == USAGE_FIELDS


def test_the_run_record_spells_exactly_the_canonical_fields() -> None:
    assert run_record.USAGE_FIELDS == USAGE_FIELDS


def test_the_run_record_imports_no_research_module() -> None:
    """Why the record's tuple is a mirror and not an import: the record writer stays light."""
    reached = {
        name
        for name in imported_modules(SOURCE_ROOT / "reporting" / "run_record.py")
        if name.startswith("noctis.research")
    }

    assert reached == set(), f"the run record reached into the research package: {sorted(reached)}"


def test_the_usage_module_names_both_mirrors_so_a_maintainer_meets_the_rule() -> None:
    doc = _prose(usage.__doc__)

    assert "pricing" in doc
    assert "run_record" in doc


# ── the one accumulator ─────────────────────────────────────────────────────────────────────
def test_one_completions_split_folds_into_the_running_totals() -> None:
    totals = _zeros()

    accumulate_usage(totals, {"input_tokens": 120, "output_tokens": 30})

    assert totals == {
        "input_tokens": 120,
        "output_tokens": 30,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def test_completions_accumulate_field_by_field() -> None:
    totals = _zeros()

    accumulate_usage(totals, dict.fromkeys(FOUR_FIELDS, 1))
    accumulate_usage(totals, dict.fromkeys(FOUR_FIELDS, 4))

    assert totals == dict.fromkeys(FOUR_FIELDS, 5)


@pytest.mark.parametrize("nothing", [None, {}], ids=["no-usage", "empty-usage"])
def test_a_client_that_reports_no_usage_contributes_nothing(nothing: dict | None) -> None:
    """The measurement floor: a fake client must never break the loop it measures."""
    totals = _zeros()

    accumulate_usage(totals, nothing)

    assert totals == _zeros()


def test_a_missing_or_null_field_contributes_zero() -> None:
    totals = _zeros()

    accumulate_usage(totals, {"input_tokens": 7, "output_tokens": None})

    assert totals == {**_zeros(), "input_tokens": 7}


def test_a_field_the_engine_does_not_bill_is_ignored() -> None:
    """A provider that reports extras adds no key: the four are what the price table bills."""
    totals = _zeros()

    accumulate_usage(totals, {"input_tokens": 2, "reasoning_tokens": 999})

    assert totals == {**_zeros(), "input_tokens": 2}


def test_a_count_that_is_not_an_int_is_read_as_one() -> None:
    totals = _zeros()

    accumulate_usage(totals, {"input_tokens": 2.7, "output_tokens": "5"})

    assert totals == {**_zeros(), "input_tokens": 2, "output_tokens": 5}


def test_the_engine_holds_exactly_one_usage_accumulator() -> None:
    """The two byte-identical copies this story deleted cannot come back one at a time."""
    found = function_definitions(SOURCE_ROOT, frozenset({"accumulate_usage", "_accumulate_usage"}))

    assert [decl.module for decl in found] == [CANONICAL], [decl.line() for decl in found]


# ── the one token estimator ─────────────────────────────────────────────────────────────────
def test_the_estimate_is_the_prefix_plus_the_serialized_history_at_four_chars_per_token() -> None:
    messages = [{"role": "user", "content": "hello"}]
    serialized = len(json.dumps(messages[0]))

    assert estimate_tokens(400, messages) == (400 + serialized) // APPROX_CHARS_PER_TOKEN


def test_the_estimate_of_a_bare_prefix_is_its_chars_over_the_ratio() -> None:
    assert estimate_tokens(4000, []) == 4000 // APPROX_CHARS_PER_TOKEN
    assert APPROX_CHARS_PER_TOKEN == 4


def test_the_estimate_is_provider_neutral_and_repeatable() -> None:
    """Deliberately independent of what any backend reports — it must work on a client that
    reports no usage at all, which is exactly when a context budget still has to hold."""
    messages = [{"role": "assistant", "content": "x" * 100}]

    assert estimate_tokens(0, messages) == estimate_tokens(0, messages)
    assert estimate_tokens(0, messages) > 0


def test_a_message_json_cannot_serialize_is_measured_not_raised() -> None:
    class Opaque:
        def __str__(self) -> str:
            return "opaque"

    assert estimate_tokens(0, [{"role": "user", "content": Opaque()}]) > 0


def test_the_engine_holds_exactly_one_token_estimator() -> None:
    found = function_definitions(SOURCE_ROOT, frozenset({"estimate_tokens", "_estimate_tokens"}))

    assert [decl.module for decl in found] == [CANONICAL], [decl.line() for decl in found]


def test_the_chars_per_token_ratio_is_declared_once() -> None:
    """One accounting means one ratio: a second copy would drift the loop off the briefings."""
    declared = [
        module
        for module, path in _sources(SOURCE_ROOT)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if any(name.endswith("APPROX_CHARS_PER_TOKEN") for name in _assigned_names(node))
    ]

    assert declared == [CANONICAL], declared


def test_the_briefing_builders_import_no_private_name() -> None:
    """The cross-import this story removed: briefings reached into the agent loop for
    ``_estimate_tokens``. A shared function is public or it is not shared."""
    imported = imported_names(SOURCE_ROOT / "research" / "briefings.py")
    private = {name for name in imported if name.startswith("_")}

    assert private == set(), f"briefings imports private names: {sorted(private)}"


def test_a_briefing_block_is_sized_with_the_shared_estimator() -> None:
    """One token accounting across the codebase, not two: what a briefing block measures is what
    the conversation loop's context budget would measure for the same text."""
    text = "a rendered briefing block\n" * 40

    assert briefings._tokens(text) == estimate_tokens(len(text), [])


# ── what the scan counts, and what it does not ──────────────────────────────────────────────
def test_a_tuple_of_the_four_fields_is_a_declaration(tmp_path: Path) -> None:
    package = _tree(
        tmp_path,
        {
            "research/episode.py": (
                "_USAGE_FIELDS = (\n"
                '    "input_tokens",\n'
                '    "output_tokens",\n'
                '    "cache_creation_input_tokens",\n'
                '    "cache_read_input_tokens",\n'
                ")\n"
            )
        },
    )

    (decl,) = field_declarations(package)

    assert decl.module == "noctis.research.episode"
    assert decl.name == "_USAGE_FIELDS"
    assert decl.shape == LIST
    assert decl.path == "noctis/research/episode.py"
    assert decl.lineno == 1


def test_a_dict_keyed_by_the_four_fields_is_a_declaration(tmp_path: Path) -> None:
    """The price table's shape: the keys are the field list, whatever the values carry."""
    package = _tree(
        tmp_path,
        {
            "research/pricing.py": (
                "USAGE_FIELDS = {\n"
                '    "input_tokens": "input_usd_per_mtok",\n'
                '    "output_tokens": "output_usd_per_mtok",\n'
                '    "cache_creation_input_tokens": "cache_write_usd_per_mtok",\n'
                '    "cache_read_input_tokens": "cache_read_usd_per_mtok",\n'
                "}\n"
            )
        },
    )

    (decl,) = field_declarations(package)

    assert decl.module == "noctis.research.pricing"
    assert decl.shape == MAPPING
    assert field_lists(package) == ()


def test_prose_that_lists_the_four_fields_is_not_a_declaration(tmp_path: Path) -> None:
    """Every module that touches usage explains the four fields. Explaining a rule must never
    break it — which is why the scan reads assignments and never prose."""
    package = _tree(
        tmp_path,
        {
            "research/ledger.py": (
                '"""Journals input_tokens, output_tokens, cache_creation_input_tokens and\n'
                'cache_read_input_tokens."""\n'
                "\n"
                "# tuple: input_tokens, output_tokens, cache_creation_input_tokens,\n"
                "#        cache_read_input_tokens\n"
                "from noctis.research.usage import USAGE_FIELDS\n"
            )
        },
    )

    assert field_declarations(package) == ()


def test_importing_the_canonical_list_under_a_new_name_is_not_a_declaration(tmp_path: Path) -> None:
    """The ledger keeps its public ``USAGE_FIELDS``; the declaration behind it lives once."""
    package = _tree(
        tmp_path,
        {"research/ledger.py": "from noctis.research.usage import USAGE_FIELDS as USAGE_FIELDS\n"},
    )

    assert field_declarations(package) == ()


def test_a_partial_list_is_not_the_field_list(tmp_path: Path) -> None:
    package = _tree(
        tmp_path,
        {"research/agent.py": '_SOME = ("input_tokens", "output_tokens")\n'},
    )

    assert field_declarations(package) == ()


def test_a_second_declaration_anywhere_is_found_and_reported(tmp_path: Path) -> None:
    package = _tree(
        tmp_path,
        {
            "research/usage.py": f"USAGE_FIELDS = {FOUR_FIELDS!r}\n",
            "engine/runtime.py": f"FIELDS = {list(FOUR_FIELDS)!r}\n",
        },
    )

    found = field_declarations(package)

    assert [decl.module for decl in found] == ["noctis.engine.runtime", "noctis.research.usage"]
    assert found == field_declarations(package)
    rendered = report(found)
    assert "noctis.engine.runtime.FIELDS" in rendered
    assert CANONICAL in rendered


def test_the_real_tree_scan_reads_every_module_that_touches_usage() -> None:
    """A scan that skipped the estimator's consumers or either mirror would pass any tree."""
    modules = {module for module, _ in _sources(SOURCE_ROOT)}

    assert {
        "noctis.reporting.run_record",
        "noctis.research.agent",
        "noctis.research.briefings",
        "noctis.research.episode",
        "noctis.research.ledger",
        "noctis.research.pricing",
        "noctis.research.usage",
    } <= modules
