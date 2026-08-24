"""One writer per research store, declared in prose and pinned here (story #331, epic #326).

Two durable stores hold what a research session knows, and they are **not** two copies of one
thing:

* the **experiment journal** (:mod:`noctis.research.journal`) holds the per-strategy *facts* — the
  thesis of an authored strategy, its class tag, every trial, the scorecard, the gate's verdict —
  and the research toolbox (:mod:`noctis.research.tools`) is the only module that writes it;
* the **session ledger** (:mod:`noctis.research.ledger`) holds one session's *narrative* — every
  thesis proposed, the stages, the episodes, the model's own verdict and its lesson, the rollup —
  and the episodic driver (:mod:`noctis.research.driver`) is the only module that writes it.

The epic's whole point is that a reported number has **one** derivation, which only holds while a
fact has one owner. So this file has three halves:

* **The rule is written down.** The two store modules, the two writer modules, the glossary and
  ``docs/research.md`` each state the rule, so a maintainer meets it before the code does — and the
  thesis **double write** is stated as deliberate, since it is the one fact both stores hold and
  therefore the one a well-meaning cleanup would "fix" into a regression.
* **A second writer cannot appear quietly.** A static scan over ``src/noctis`` classifies every
  call that writes either store and names any that comes from a module the store does not belong
  to. The writer method names are read off the classes themselves, so a new ``record_*`` fact joins
  the guard the moment it is declared.
* **``undecided`` keeps its one meaning** — names with no verdict spent yet — in the counters
  surface and in the parity harness's row.

**The scan reads code, never prose.** Each file is tokenized and every comment and string literal
dropped (:func:`tests.test_toolbox_boundary.code_text`, one spelling of that idiom for both
guards), so the paragraphs above may quote ``ledger.record_thesis(...)`` as the thing being
described without tripping the guard they describe.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path

import pytest

from noctis.research import driver, journal, ledger, parity, tools
from noctis.research.journal import ExperimentJournal
from noctis.research.ledger import SessionLedger
from noctis.research.surface import SessionCounters
from tests.test_toolbox_boundary import code_text

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "noctis"


def _record_methods(store: type) -> frozenset[str]:
    """The store's write surface, read off the class: every ``record_*`` method it declares.

    Introspected rather than listed, so a new fact cannot gain a writer the scan never heard of:
    declaring ``record_whatever`` on either class puts it under the guard in the same commit.
    """
    return frozenset(name for name in vars(store) if name.startswith("record_"))


@dataclass(frozen=True)
class Store:
    """One durable store, the single module entitled to write it, and how a write is spelled."""

    #: The store in the glossary's words, for the report a violation prints.
    name: str
    #: The module that owns the record schema, relative to the package root.
    home: str
    #: The **one** writer, relative to the package root. A second entry is a design decision.
    writer: str
    #: How the store is spelled at a call site — the receiver token of ``<recv>.record_x(...)``.
    receivers: tuple[str, ...]
    #: The write surface, introspected from the class.
    writers: frozenset[str]

    @property
    def allowed(self) -> frozenset[str]:
        """The modules a write may come from: the one writer, and the store's own module."""
        return frozenset({self.writer, self.home})


JOURNAL = Store(
    name="experiment journal",
    home="research.journal",
    writer="research.tools",
    receivers=("journal",),
    writers=_record_methods(ExperimentJournal),
)

LEDGER = Store(
    name="session ledger",
    home="research.ledger",
    writer="research.driver",
    receivers=("ledger",),
    writers=_record_methods(SessionLedger),
)

STORES = (JOURNAL, LEDGER)

#: A write: an attribute call of a ``record_*`` method on some receiver. Which store it writes is
#: decided afterwards by the method name (and, for the one name both stores declare, the receiver),
#: so a call spelled on the wrong store — ``ledger.record_trial(...)`` — is still judged as what it
#: actually writes. The optional leading name is captured only so a report quotes the call as its
#: author typed it (``self.journal.record_trial``, not ``journal.record_trial``).
WRITE = re.compile(r"\b(?:(\w+)\s*\.\s*)?(\w+)\s*\.\s*(record_\w+)\s*\(")

# Restated in every report, because a violation is most usefully answered by the rule it broke.
_CONTRACT = (
    "one writer per store: the toolbox writes the experiment journal, the episodic driver writes "
    "the session ledger, and neither store mirrors the other"
)


@dataclass(frozen=True)
class StoreWrite:
    """One write of one store: what it wrote, who wrote it, and exactly where."""

    store: Store
    module: str
    rel: str
    path: str
    lineno: int
    call: str

    @property
    def stray(self) -> bool:
        """True when the writing module is not the store's writer or the store's own module."""
        return self.rel not in self.store.allowed

    def line(self) -> str:
        """The single line a failure prints — the module first, since that is what must change."""
        return (
            f"{self.module} writes the {self.store.name} via {self.call} "
            f"({self.path}:{self.lineno}) — its one writer is noctis.{self.store.writer}"
        )


def store_writes(package_root: Path) -> tuple[StoreWrite, ...]:
    """Every call under ``package_root`` that writes one of the two stores, in a fixed order.

    ``package_root`` is the top-level package directory (``src/noctis`` for this repo, or a
    fabricated one in a test); dotted module names are derived from the root's own name, so a
    miniature tree reads exactly like the real one. Sorted by module, then line, then call: the
    same tree always yields the same tuple, so a failure message is stable across machines.
    """
    root = Path(package_root)
    found: list[StoreWrite] = []
    for module, rel, path in _sources(root):
        text = code_text(path.read_text(encoding="utf-8"))
        for match in WRITE.finditer(text):
            _lead, receiver, method = match.groups()
            for store in _stores_written(receiver, method):
                found.append(
                    StoreWrite(
                        store=store,
                        module=module,
                        rel=rel,
                        path=path.relative_to(root.parent).as_posix(),
                        lineno=text.count("\n", 0, match.start()) + 1,
                        call=_call(match.group()),
                    )
                )
    return tuple(sorted(found, key=lambda write: (write.module, write.lineno, write.call)))


def stray_writes(package_root: Path) -> tuple[StoreWrite, ...]:
    """The writes that came from a module the store they write does not belong to."""
    return tuple(write for write in store_writes(package_root) if write.stray)


def writers_by_store(package_root: Path) -> dict[str, set[str]]:
    """Which modules write each store — the answer that makes a green verdict a verdict."""
    found: defaultdict[str, set[str]] = defaultdict(set)
    for write in store_writes(package_root):
        found[write.store.name].add(write.module)
    return dict(found)


def report(strays: tuple[StoreWrite, ...]) -> str:
    """The verdict a failing guard prints: every offending module, then the rule they broke."""
    if not strays:
        return f"every research store has exactly one writer: {_CONTRACT}"
    header = f"a research store is written from {len(strays)} module(s) that do not own it:"
    return "\n".join([header, *(f"  {write.line()}" for write in strays), f"  {_CONTRACT}"])


def _stores_written(receiver: str, method: str) -> tuple[Store, ...]:
    """The store(s) a ``<receiver>.<method>(`` call writes — empty when it writes neither.

    The method name decides, because that is what the store actually declares: a ``record_*`` on
    some other object (a memory store's ``record_rejected``) is nobody's business here. The
    receiver only disambiguates the one name **both** stores declare, ``record_thesis``; a
    receiver spelled as neither store is judged against both, since a write the guard cannot
    attribute is exactly the kind that should have to be argued for.
    """
    owners = tuple(store for store in STORES if method in store.writers)
    if len(owners) < 2:
        return owners
    named = tuple(store for store in owners if receiver in store.receivers)
    return named or owners


def _call(matched: str) -> str:
    """A matched write as its author typed it, with the tokenizer's spacing removed."""
    return f"{''.join(matched.split())}…)"


def _sources(root: Path) -> tuple[tuple[str, str, Path], ...]:
    """The package's modules as ``(dotted name, name relative to the root, file)``, in order."""
    found = ((_module_name(root, path), path) for path in root.rglob("*.py"))
    return tuple(sorted((name, name.removeprefix(f"{root.name}."), path) for name, path in found))


def _module_name(root: Path, path: Path) -> str:
    """The dotted name of the module at ``path``, rooted at the package directory's own name."""
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
    """One docstring with its line wrapping collapsed, so reflowing a paragraph is not a change."""
    return " ".join((text or "").split())


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _glossary_entry(title: str) -> str:
    """One ``## <title>`` section of the domain glossary."""
    text = _read("CONTEXT.md")
    start = text.find(f"## {title}\n")
    assert start >= 0, f"CONTEXT.md: the '{title}' entry is gone — retarget this test"
    end = text.find("\n## ", start + 1)
    return text[start : end if end >= 0 else len(text)]


# ── the real tree: one writer per store ─────────────────────────────────────────────────────
def test_no_module_writes_a_store_it_does_not_own() -> None:
    strays = stray_writes(SOURCE_ROOT)

    assert strays == (), report(strays)


def test_the_scan_finds_the_two_real_writers_and_nobody_else() -> None:
    """What makes the green above a verdict rather than an empty walk: the two writers are found,
    each writing its own store and only its own store."""
    found = writers_by_store(SOURCE_ROOT)

    assert found == {
        JOURNAL.name: {"noctis.research.tools"},
        LEDGER.name: {"noctis.research.driver"},
    }


@pytest.mark.parametrize("store", STORES, ids=lambda store: store.name)
def test_every_record_method_a_store_declares_is_under_the_guard(store: Store) -> None:
    """The write surface is introspected, so declaring a new fact enrols it in the same commit."""
    assert store.writers
    assert "record_thesis" in store.writers  # the one name both stores declare


def test_the_real_tree_scan_reads_every_module_that_holds_a_store() -> None:
    """A scan that skipped a store's module, its writer, or the composition root that hands the
    store over would pass any tree ever written."""
    modules = {module for module, _, _ in _sources(SOURCE_ROOT)}

    assert {
        "noctis.bootstrap",
        "noctis.eval.decide_site",
        "noctis.reporting.run_tree.evidence",
        "noctis.research.briefings",
        "noctis.research.driver",
        "noctis.research.journal",
        "noctis.research.ledger",
        "noctis.research.tools",
    } <= modules


# ── what counts as a write ──────────────────────────────────────────────────────────────────
def test_a_module_that_journals_outside_the_toolbox_is_named(tmp_path: Path) -> None:
    package = _tree(
        tmp_path,
        {"research/driver.py": "def go(journal, name):\n    journal.record_trial(name)\n"},
    )

    (write,) = stray_writes(package)

    assert write.module == "noctis.research.driver"
    assert write.store is JOURNAL
    assert write.path == "noctis/research/driver.py"
    assert write.lineno == 2
    assert write.call == "journal.record_trial(…)"


def test_a_module_that_ledgers_outside_the_driver_is_named(tmp_path: Path) -> None:
    package = _tree(
        tmp_path,
        {"research/tools.py": "def go(self, name):\n    self.ledger.record_verdict(name)\n"},
    )

    (write,) = stray_writes(package)

    assert write.module == "noctis.research.tools"
    assert write.store is LEDGER
    assert write.call == "self.ledger.record_verdict(…)"


def test_a_store_written_through_the_other_stores_receiver_is_still_judged_as_itself(
    tmp_path: Path,
) -> None:
    """Naming a journal ``ledger`` does not make a journal write a ledger write: the method the
    store declares is what a call actually writes."""
    package = _tree(
        tmp_path,
        {"research/driver.py": "def go(ledger, name):\n    ledger.record_scorecard(name)\n"},
    )

    (write,) = stray_writes(package)

    assert write.store is JOURNAL


def test_the_shared_thesis_write_is_attributed_by_its_receiver(tmp_path: Path) -> None:
    """``record_thesis`` is the deliberate double write — the one name both stores declare — so the
    receiver decides which store a call wrote, and each writer keeps its own half."""
    package = _tree(
        tmp_path,
        {
            "research/driver.py": "def go(ledger):\n    ledger.record_thesis('a')\n",
            "research/tools.py": "def go(self):\n    self.journal.record_thesis('a')\n",
        },
    )

    assert stray_writes(package) == ()
    assert writers_by_store(package) == {
        JOURNAL.name: {"noctis.research.tools"},
        LEDGER.name: {"noctis.research.driver"},
    }


def test_the_shared_thesis_write_from_the_wrong_module_is_named(tmp_path: Path) -> None:
    package = _tree(
        tmp_path,
        {"research/driver.py": "def go(self):\n    self.journal.record_thesis('a')\n"},
    )

    (write,) = stray_writes(package)

    assert write.store is JOURNAL
    assert write.module == "noctis.research.driver"


def test_a_shared_write_the_guard_cannot_attribute_is_judged_against_both_stores(
    tmp_path: Path,
) -> None:
    """An unfamiliar receiver is not a loophole: a ``record_thesis`` nobody can attribute is judged
    against both stores, so it is stray unless it is argued for here."""
    package = _tree(
        tmp_path, {"engine/runtime.py": "def go(store):\n    store.record_thesis('a')\n"}
    )

    strays = stray_writes(package)

    assert {write.store for write in strays} == {JOURNAL, LEDGER}


# ── what does not count ─────────────────────────────────────────────────────────────────────
def test_each_store_may_be_written_by_its_one_writer(tmp_path: Path) -> None:
    package = _tree(
        tmp_path,
        {
            "research/tools.py": "def go(self, n):\n    self.journal.record_trial(n)\n",
            "research/driver.py": "def go(ledger, n):\n    ledger.record_stage(n)\n",
        },
    )

    assert stray_writes(package) == ()


def test_a_store_may_write_its_own_record(tmp_path: Path) -> None:
    """The store's own module owns the schema, so a record it writes itself is the seam working."""
    package = _tree(
        tmp_path,
        {"research/ledger.py": "def go(self, n):\n    self.ledger.record_episode(n)\n"},
    )

    assert stray_writes(package) == ()


def test_a_record_method_on_something_that_is_not_a_store_is_not_a_write(tmp_path: Path) -> None:
    """The memory store's ``record_rejected`` is nobody's business here — the scan is about the two
    research stores, named by the methods they actually declare."""
    package = _tree(
        tmp_path,
        {"engine/research.py": "def go(memory, f, p):\n    memory.record_rejected(f, p)\n"},
    )

    assert store_writes(package) == ()


def test_reading_a_store_is_never_a_write(tmp_path: Path) -> None:
    package = _tree(
        tmp_path,
        {
            "reporting/evidence.py": (
                "def go(ledger, journal, n):\n"
                "    return ledger.rollup(), ledger.undecided_names(), journal.trials(n)\n"
            )
        },
    )

    assert store_writes(package) == ()


def test_prose_that_quotes_a_write_is_not_a_write(tmp_path: Path) -> None:
    """Both store docstrings describe the writes they own. Explaining a boundary must never break
    it — which is why the scan reads code and never prose."""
    package = _tree(
        tmp_path,
        {
            "research/briefings.py": (
                '"""The driver calls ledger.record_thesis(line) for every thesis proposed."""\n'
                "\n"
                "# the toolbox calls journal.record_trial(name, params) per trial\n"
                "OLD = 'journal.record_verdict(name)'\n"
            )
        },
    )

    assert store_writes(package) == ()


# ── the verdict ─────────────────────────────────────────────────────────────────────────────
def test_the_scan_reports_strays_in_a_deterministic_order(tmp_path: Path) -> None:
    package = _tree(
        tmp_path,
        {
            "research/driver.py": "def go(journal, n):\n    journal.record_trial(n)\n",
            "cli.py": "def go(ledger, n):\n    ledger.record_verdict(n)\n",
        },
    )

    first = stray_writes(package)
    second = stray_writes(package)

    assert [write.module for write in first] == ["noctis.cli", "noctis.research.driver"]
    assert first == second


def test_the_report_names_every_offending_module_and_the_rule_it_broke(tmp_path: Path) -> None:
    package = _tree(
        tmp_path,
        {
            "research/driver.py": "def go(journal, n):\n    journal.record_trial(n)\n",
            "cli.py": "def go(ledger, n):\n    ledger.record_verdict(n)\n",
        },
    )

    rendered = report(stray_writes(package))

    assert "noctis.research.driver" in rendered
    assert "noctis.cli" in rendered
    assert "one writer per store" in rendered


def test_a_write_renders_as_one_line_naming_module_call_position_and_the_one_writer() -> None:
    write = StoreWrite(
        store=JOURNAL,
        module="noctis.research.driver",
        rel="research.driver",
        path="noctis/research/driver.py",
        lineno=12,
        call="journal.record_trial(…)",
    )

    assert write.line() == (
        "noctis.research.driver writes the experiment journal via journal.record_trial(…) "
        "(noctis/research/driver.py:12) — its one writer is noctis.research.tools"
    )


def test_the_allowlist_is_one_writer_plus_the_stores_own_module() -> None:
    """Two entries per store, and widening either is a design decision somebody has to type."""
    assert JOURNAL.allowed == frozenset({"research.tools", "research.journal"})
    assert LEDGER.allowed == frozenset({"research.driver", "research.ledger"})


# ── the rule, written down ──────────────────────────────────────────────────────────────────
def test_the_journal_docstring_names_its_one_writer_and_refuses_to_be_a_mirror() -> None:
    doc = _prose(journal.__doc__)

    assert "noctis.research.tools" in doc
    assert "noctis.research.ledger" in doc
    assert "not a mirror" in doc


def test_the_ledger_docstring_names_its_one_writer_and_refuses_to_be_a_mirror() -> None:
    doc = _prose(ledger.__doc__)

    assert "noctis.research.driver" in doc
    assert "noctis.research.journal" in doc
    assert "not a mirror" in doc


def test_both_store_docstrings_state_the_thesis_double_write_as_deliberate() -> None:
    """The one fact both stores hold, and therefore the one a cleanup would "fix" into a
    regression: the ledger carries every thesis *proposed*, the journal only an *authored* name."""
    for doc in (_prose(journal.__doc__), _prose(ledger.__doc__)):
        assert "double write" in doc

    assert "phantom" in _prose(journal.__doc__)
    assert "failed AUTHOR" in _prose(ledger.__doc__)


def test_the_toolbox_docstring_states_its_side_of_the_rule() -> None:
    assert "the journal's one writer" in _prose(tools.__doc__)


def test_the_driver_docstring_states_its_side_of_the_rule() -> None:
    assert "the ledger's one writer" in _prose(driver.__doc__)


def test_the_glossary_defines_the_session_ledger_beside_the_experiment_journal() -> None:
    text = _read("CONTEXT.md")
    experiments = text.find("## Experiment journal\n")
    sessions = text.find("## Session ledger\n")

    assert experiments >= 0, "CONTEXT.md: the 'Experiment journal' entry is gone — retarget this"
    assert sessions > experiments, "CONTEXT.md has no 'Session ledger' entry after the journal's"


def test_the_session_ledger_entry_says_what_it_is_who_writes_it_and_when_that_was_decided() -> None:
    entry = _glossary_entry("Session ledger")

    assert "state/sessions/" in entry
    assert "noctis.research.ledger" in entry
    assert "noctis.research.driver" in entry
    assert "decided 2026-08-24" in entry


def test_the_experiment_journal_entry_names_its_one_writer_and_points_at_the_ledger() -> None:
    entry = _glossary_entry("Experiment journal")

    assert "noctis.research.tools" in entry
    assert "session ledger" in entry


def test_the_research_page_states_the_ownership_rule() -> None:
    """A reader of the docs meets the rule where the two stores are discussed, not only in code."""
    page = _read("docs/research.md")
    start = page.find("### Two stores, one writer each\n")
    assert start >= 0, "docs/research.md has no 'Two stores, one writer each' section"
    section = page[start : page.find("\n## ", start + 1)]

    assert "noctis.research.tools" in section
    assert "noctis.research.driver" in section
    assert "no verdict spent" in section


# ── undecided keeps its one meaning ─────────────────────────────────────────────────────────
def test_the_counters_snapshot_keeps_the_undecided_field_and_states_what_it_counts() -> None:
    """The field name is load-bearing (every reader of the surface holds it); its *meaning* is the
    ledger's — a name with no verdict spent on it yet."""
    assert "undecided" in {field.name for field in fields(SessionCounters)}
    assert "no verdict spent" in _prose(SessionCounters.__doc__)


def test_the_parity_row_still_reads_undecided_as_authored_but_never_decided() -> None:
    """US13's row, now exactly true: a spent verdict settles a candidate whatever it said.

    Read with its line wrapping collapsed, so reflowing the paragraph is not a semantic change.
    """
    assert "authored but never carried to a verdict" in _prose(parity.__doc__)
