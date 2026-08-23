"""The research-toolbox boundary (epic #255, story #262): nobody reaches through the seam.

:mod:`noctis.research.surface` declares what a reader of the research toolbox may ask for —
:class:`~noctis.research.surface.ResearchFacts` for a renderer, and
:class:`~noctis.research.surface.Toolbox` for a driver. This file is the check that the source tree
still honours it, and it has two halves, because a seam is only held by both:

* **Nothing reaches around it.** A static scan over ``src/noctis`` returns every module outside
  :mod:`noctis.research.tools` that reads a *collaborator* off a toolbox (the journal, the lake,
  the registry, the memory, the live counters, the loose limit scalars) or probes one with
  ``getattr``. Those are the fifteen ``toolbox: Any`` annotations and twelve ``getattr(toolbox, …)``
  defaults the epic deleted; this is what keeps them deleted.
* **The seam is what the objects answer.** Four objects are measured against the Protocol they
  claim: the production toolbox, the episodic driver's fake, and the eval layer's neutral session
  and case toolbox. A boundary nothing conforms to is a boundary that has quietly moved.

**The scan reads code, never prose.** Each file is tokenized and every comment and string literal
is dropped before the patterns run, so an explanation may quote the reach it replaced —
:mod:`noctis.research.surface`'s own opening paragraph quotes ``toolbox.journal`` and
``toolbox.lake.preflight.budget_usd`` as the thing that used to happen — without tripping the
guard. A boundary that made contributors reword their prose would be reworded out of existence.

The guard lives here rather than in the package (unlike ``noctis.eval.guard``, whose layer states
its own contract to production code) because nothing in production reads this verdict: the test
*is* the enforcement. Shape and voice: ``tests/test_eval_boundary.py``.
"""

from __future__ import annotations

import io
import re
import tokenize
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest

from noctis.eval.decide_site import NEUTRAL_SESSION, decide_input
from noctis.research.surface import ResearchFacts, Toolbox
from tests.test_episodic_driver import FakeToolbox
from tests.test_eval_decide_site import WINDOW, _approved_case
from tests.test_research_tools import _make_toolbox

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src" / "noctis"

#: The one module allowed to name the collaborators, spelled relative to the package root so a
#: fabricated tree in a test reads exactly like the real one. It is the module that *owns* them:
#: a toolbox holding a journal is the seam working. A second entry here is a design decision.
ALLOWED = frozenset({"research.tools"})

#: The collaborators a reader used to reach *through* the toolbox for. Every one of them is still
#: a plain public attribute on :class:`~noctis.research.tools.ResearchToolbox` — "private" means
#: "not on the surface", not an underscore rename — so only this scan keeps them off the seam.
COLLABORATORS = (
    "capture",
    "exhausted",
    "families",
    "journal",
    "lake",
    "mandate_source",
    "memory",
    "registry",
    "settings",
    "strategies_dir",
)

#: The live session counters. They are mutable attributes the toolbox's own tools bump, which is
#: why a reader takes ``session_counters()`` — a frozen snapshot — instead of a view of them.
COUNTERS = (
    "author_calls",
    "backtests_run",
    "escalations",
    "promotions",
    "rejections",
    "strategies_touched",
    "undecided",
)

#: The four ceilings as loose scalars. They travel as one frozen ``limits`` value, so a briefing
#: can never show one session's backtest ceiling beside another's trial floor.
LIMIT_SCALARS = ("default_sweep_trials", "max_author_calls", "max_backtests", "min_trials")

#: How a toolbox is spelled at the point of the reach. ``self.toolbox`` and ``session.toolbox``
#: need no entry of their own: the receiver of the reached attribute is the ``toolbox`` token
#: either way, and matching on whole tokens is what makes that true.
RECEIVERS = ("box", "toolbox")

_MEMBERS = "|".join(sorted(COLLABORATORS + COUNTERS + LIMIT_SCALARS))

#: A collaborator read off a toolbox. Word-anchored on both ends, so ``toolbox.memory_tail`` and
#: ``toolbox.journal_evidence`` — surface members, and exactly what a reader should call — are not
#: matched by ``memory`` and ``journal``. The optional leading name is captured only so a failure
#: quotes the reach as its author typed it (``self.toolbox.memory``, not ``toolbox.memory``).
REACH = re.compile(rf"\b(?:\w+\s*\.\s*)?(?:{'|'.join(RECEIVERS)})\s*\.\s*(?:{_MEMBERS})\b")

#: A ``getattr`` probe aimed at a toolbox, under any spelling of it. The default argument is the
#: reason this is banned outright rather than by member name: a probe that misses answers with an
#: invented fact, which is how a briefing came to state a budget nobody configured.
PROBE = re.compile(r"\bgetattr\s*\(\s*[\w.\s]*\btoolbox\b")

# Restated in every report, because a violation is most usefully answered by the rule it broke.
_CONTRACT = (
    "a reader of the toolbox holds noctis.research.surface — ResearchFacts to render, Toolbox to "
    "drive — and reads facts, never a collaborator"
)

# The tokens that are *code*: everything else (comments, docstrings, string literals) is prose the
# scan never reads. NUMBER is kept only so a line's text stays recognisable to a human reading a
# failure; no pattern here matches a literal.
_CODE_TOKENS = frozenset({tokenize.NAME, tokenize.OP, tokenize.NUMBER})


@dataclass(frozen=True)
class ReachThrough:
    """One reach around the surface: who reached, for what, and exactly where."""

    module: str
    path: str
    lineno: int
    reach: str

    def line(self) -> str:
        """The single line a failure prints — the module first, since that is what must change."""
        return f"{self.module} reaches {self.reach} ({self.path}:{self.lineno})"


def code_text(source: str) -> str:
    """``source`` with every comment and string literal dropped, line numbering preserved.

    Only ``NAME``/``OP``/``NUMBER`` tokens survive, joined by single spaces and left on their own
    lines, so a match still names the line a reader would open. Two consequences are deliberate:
    prose is invisible (a docstring may quote the reach it replaced), and a reach spelled across
    two lines is still one match, reported at the line it starts on.
    """
    rows: defaultdict[int, list[str]] = defaultdict(list)
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in _CODE_TOKENS:
            rows[tok.start[0]].append(tok.string)
    if not rows:
        return ""
    return "\n".join(" ".join(rows.get(row, ())) for row in range(1, max(rows) + 1))


def toolbox_reach_throughs(package_root: Path) -> tuple[ReachThrough, ...]:
    """Every module under ``package_root`` outside :data:`ALLOWED` that reaches around the surface.

    ``package_root`` is the top-level package directory (``src/noctis`` for this repo, or a
    fabricated one in a test); dotted module names are derived from the root's own name, so a
    miniature tree reads exactly like the real one.

    Sorted by module, then line, then expression: the same tree always yields the same tuple, so a
    failure message is stable across machines and runs.
    """
    root = Path(package_root)
    found: list[ReachThrough] = []
    for module, path in _sources(root):
        text = code_text(path.read_text(encoding="utf-8"))
        for pattern, render in ((REACH, _read), (PROBE, _probe)):
            for match in pattern.finditer(text):
                found.append(
                    ReachThrough(
                        module=module,
                        path=path.relative_to(root.parent).as_posix(),
                        lineno=text.count("\n", 0, match.start()) + 1,
                        reach=render(match.group()),
                    )
                )
    return tuple(sorted(found, key=lambda item: (item.module, item.lineno, item.reach)))


def scanned_modules(package_root: Path) -> tuple[str, ...]:
    """Every module the scan judges — the package's own ``*.py`` files, minus :data:`ALLOWED`.

    Stated as its own answer so a green verdict can be shown to be a verdict: a scan that walked
    nothing would pass every tree ever written.
    """
    return tuple(module for module, _ in _sources(Path(package_root)))


def report(reaches: tuple[ReachThrough, ...]) -> str:
    """The verdict a failing guard prints: every offending module, then the rule they broke."""
    if not reaches:
        return f"nothing reaches around the toolbox surface: {_CONTRACT}"
    header = f"the research-toolbox surface is reached around {len(reaches)} time(s):"
    return "\n".join([header, *(f"  {reach.line()}" for reach in reaches), f"  {_CONTRACT}"])


def _read(matched: str) -> str:
    """A matched reach as a reader would have typed it, with the tokenizer's spacing removed."""
    return "".join(matched.split())


def _probe(matched: str) -> str:
    """A matched probe, closed off: the arguments after the toolbox are not what makes it one."""
    return f"{_read(matched)}, …)"


def _sources(root: Path) -> tuple[tuple[str, Path], ...]:
    """The package's modules as ``(dotted name, file)``, allowlist applied, in a fixed order."""
    found = (
        (_module_name(root, path), path)
        for path in root.rglob("*.py")
        if _module_name(root, path).removeprefix(f"{root.name}.") not in ALLOWED
    )
    return tuple(sorted(found))


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


# ── the real tree ───────────────────────────────────────────────────────────────────────────
def test_the_real_source_tree_reaches_through_no_toolbox() -> None:
    reaches = toolbox_reach_throughs(SOURCE_ROOT)

    assert reaches == (), report(reaches)


def test_the_real_tree_scan_reads_every_reader_of_the_toolbox_and_skips_its_owner() -> None:
    """What makes the green above a verdict rather than an empty walk: the modules the epic
    re-pointed at the surface are all in the scan, and the module that owns the collaborators is
    the only one out of it."""
    modules = set(scanned_modules(SOURCE_ROOT))

    assert "noctis.research.tools" not in modules
    assert {
        "noctis.bootstrap",
        "noctis.cli",
        "noctis.engine.runtime",
        "noctis.eval.decide_site",
        "noctis.eval.episodic_sites",
        "noctis.research.agent",
        "noctis.research.briefings",
        "noctis.research.digests",
        "noctis.research.driver",
        "noctis.research.prompt",
        "noctis.research.surface",
    } <= modules


# ── what counts as a reach ──────────────────────────────────────────────────────────────────
def test_a_module_that_reads_a_collaborator_off_the_toolbox_is_named(tmp_path: Path) -> None:
    package = _tree(
        tmp_path,
        {"research/briefings.py": "def go(toolbox):\n    return toolbox.journal.records(name)\n"},
    )

    reaches = toolbox_reach_throughs(package)

    assert [reach.module for reach in reaches] == ["noctis.research.briefings"]


def test_the_reach_names_the_file_the_line_and_the_expression(tmp_path: Path) -> None:
    package = _tree(
        tmp_path,
        {"research/briefings.py": "import json\n\n\ndef go(toolbox):\n    return toolbox.lake\n"},
    )

    (reach,) = toolbox_reach_throughs(package)

    assert reach.path == "noctis/research/briefings.py"
    assert reach.lineno == 5
    assert reach.reach == "toolbox.lake"


def test_a_getattr_probe_at_the_toolbox_is_a_reach(tmp_path: Path) -> None:
    package = _tree(
        tmp_path,
        {"research/prompt.py": 'def go(toolbox):\n    return getattr(toolbox, "lake", None)\n'},
    )

    (reach,) = toolbox_reach_throughs(package)

    assert reach.module == "noctis.research.prompt"
    assert reach.lineno == 2
    assert reach.reach == "getattr(toolbox, …)"


def test_a_probe_at_a_toolbox_held_as_an_attribute_is_a_reach(tmp_path: Path) -> None:
    """The default is the whole problem: a probe that misses invents a fact nobody measured."""
    package = _tree(
        tmp_path,
        {"cli.py": 'def go(self):\n    return getattr(self.toolbox, "promotions", 0)\n'},
    )

    (reach,) = toolbox_reach_throughs(package)

    assert reach.reach == "getattr(self.toolbox, …)"


def test_a_probe_that_spans_two_lines_is_still_named_at_its_own_line(tmp_path: Path) -> None:
    wrapped = """def go(session):
    return getattr(
        session.toolbox,
        "min_trials",
        0,
    )
"""
    package = _tree(tmp_path, {"cli.py": wrapped})

    (reach,) = toolbox_reach_throughs(package)

    assert reach.lineno == 2


@pytest.mark.parametrize(
    "expression",
    [
        "toolbox.journal",
        "self.toolbox.memory",
        "session.toolbox.registry",
        "box.lake",
        "toolbox.families",
        "toolbox.settings",
        "toolbox.strategies_dir",
        "toolbox.exhausted",
        "toolbox.capture",
        "toolbox.mandate_source",
    ],
)
def test_every_spelling_of_a_collaborator_reach_is_counted(tmp_path: Path, expression: str) -> None:
    package = _tree(
        tmp_path,
        {"research/driver.py": f"def go(session, self, toolbox, box):\n    return {expression}\n"},
    )

    reaches = toolbox_reach_throughs(package)

    assert [reach.reach for reach in reaches] == [expression]


@pytest.mark.parametrize(
    "member",
    [
        "promotions",
        "rejections",
        "author_calls",
        "backtests_run",
        "escalations",
        "strategies_touched",
        "undecided",
    ],
)
def test_a_live_counter_read_off_the_toolbox_is_a_reach(tmp_path: Path, member: str) -> None:
    """The counters are a snapshot (``session_counters()``), never a view that mutates later."""
    package = _tree(
        tmp_path, {"engine/runtime.py": f"def go(toolbox):\n    return toolbox.{member}\n"}
    )

    assert [reach.reach for reach in toolbox_reach_throughs(package)] == [f"toolbox.{member}"]


@pytest.mark.parametrize(
    "member", ["min_trials", "max_backtests", "default_sweep_trials", "max_author_calls"]
)
def test_a_loose_limit_scalar_read_off_the_toolbox_is_a_reach(tmp_path: Path, member: str) -> None:
    """The four ceilings travel as one frozen ``limits`` value, so no reader assembles its own."""
    package = _tree(
        tmp_path, {"research/agent.py": f"def go(toolbox):\n    return toolbox.{member}\n"}
    )

    assert [reach.reach for reach in toolbox_reach_throughs(package)] == [f"toolbox.{member}"]


# ── what does not count ─────────────────────────────────────────────────────────────────────
def test_the_toolbox_module_may_read_its_own_collaborators(tmp_path: Path) -> None:
    """The allowlist is the one module that *owns* them: a toolbox holding a journal is the seam
    working, not a breach of it."""
    package = _tree(
        tmp_path,
        {
            "research/tools.py": (
                "def go(toolbox):\n"
                "    probe = getattr(toolbox, 'lake', None)\n"
                "    return toolbox.journal, toolbox.min_trials, probe\n"
            )
        },
    )

    assert toolbox_reach_throughs(package) == ()


@pytest.mark.parametrize(
    "expression",
    [
        "toolbox.memory_tail()",
        "toolbox.journal_evidence('probe')",
        "toolbox.lake_inventory(limit=10)",
        "toolbox.limits.sweep_trials",
        "toolbox.session_counters().promotions",
        "toolbox.champion_board()",
        "toolbox.capture_episode(brief, knobs)",
        "toolbox.class_exhausted(tag)",
    ],
)
def test_a_surface_member_is_never_a_reach(tmp_path: Path, expression: str) -> None:
    """Word-anchored, so ``memory_tail`` is not ``memory`` and ``limits.sweep_trials`` is not a
    loose scalar — the surface is exactly what a reader is *supposed* to call."""
    body = f"def go(toolbox, brief, knobs, tag):\n    return {expression}\n"
    package = _tree(tmp_path, {"research/briefings.py": body})

    assert toolbox_reach_throughs(package) == ()


def test_prose_that_quotes_the_old_reach_is_not_a_violation(tmp_path: Path) -> None:
    """The surface module's own docstring names ``toolbox.journal`` as what it replaced. Explaining
    a boundary must never break it."""
    package = _tree(
        tmp_path,
        {
            "research/surface.py": (
                '"""The seam that replaced ``toolbox.journal`` and toolbox.lake."""\n'
                "\n"
                "# a reader used to write toolbox.registry.capacity here\n"
                "OLD = 'getattr(toolbox, \"memory\", None)'\n"
            )
        },
    )

    assert toolbox_reach_throughs(package) == ()


def test_a_collaborator_read_off_something_that_is_not_a_toolbox_is_not_a_reach(
    tmp_path: Path,
) -> None:
    """The scan is about one object: a session's own journal, or a settings module's own lake, is
    nobody's business but its own."""
    package = _tree(
        tmp_path,
        {"research/agent.py": "def go(session, self):\n    return session.journal, self.lake\n"},
    )

    assert toolbox_reach_throughs(package) == ()


def test_a_getattr_probe_at_something_else_is_not_a_reach(tmp_path: Path) -> None:
    package = _tree(
        tmp_path, {"research/driver.py": 'def go(fn):\n    return getattr(fn, "__name__", fn)\n'}
    )

    assert toolbox_reach_throughs(package) == ()


# ── the verdict ─────────────────────────────────────────────────────────────────────────────
def test_the_scan_reports_reaches_in_a_deterministic_order(tmp_path: Path) -> None:
    package = _tree(
        tmp_path,
        {
            "research/driver.py": "def go(toolbox):\n    return toolbox.lake\n",
            "cli.py": "def go(toolbox):\n    return toolbox.journal\n",
            "engine/runtime.py": "def go(toolbox):\n    return toolbox.promotions\n",
        },
    )

    first = toolbox_reach_throughs(package)
    second = toolbox_reach_throughs(package)

    assert [reach.module for reach in first] == [
        "noctis.cli",
        "noctis.engine.runtime",
        "noctis.research.driver",
    ]
    assert first == second


def test_the_report_names_every_offending_module_and_the_rule_it_broke(tmp_path: Path) -> None:
    package = _tree(
        tmp_path,
        {
            "research/driver.py": "def go(toolbox):\n    return toolbox.lake\n",
            "cli.py": "def go(toolbox):\n    return toolbox.journal\n",
        },
    )

    rendered = report(toolbox_reach_throughs(package))

    assert "noctis.research.driver" in rendered
    assert "noctis.cli" in rendered
    assert "noctis.research.surface" in rendered


def test_a_reach_renders_as_one_line_naming_module_expression_and_position() -> None:
    reach = ReachThrough(
        module="noctis.research.driver",
        path="noctis/research/driver.py",
        lineno=12,
        reach="toolbox.journal",
    )

    assert reach.line() == (
        "noctis.research.driver reaches toolbox.journal (noctis/research/driver.py:12)"
    )


def test_the_allowlist_names_only_the_module_that_owns_the_collaborators() -> None:
    """One entry, spelled relative to the package root, so a fabricated tree reads like the real
    one — and so widening it is a design decision somebody has to type."""
    assert ALLOWED == frozenset({"research.tools"})


# ── the seam the scan protects ──────────────────────────────────────────────────────────────
def test_the_production_toolbox_satisfies_the_driver_tier(tmp_path: Path) -> None:
    box = _make_toolbox(tmp_path)

    assert isinstance(box, Toolbox)
    assert isinstance(box, ResearchFacts)


def test_the_episodic_drivers_fake_satisfies_the_driver_tier() -> None:
    """The deterministic driver tests drive a fake; the fake is measured against the real seam."""
    assert isinstance(FakeToolbox(), Toolbox)


def test_the_benchs_neutral_session_satisfies_the_facts_tier() -> None:
    assert isinstance(NEUTRAL_SESSION, ResearchFacts)
    assert not isinstance(NEUTRAL_SESSION, Toolbox)


def test_the_benchs_case_toolbox_satisfies_the_facts_tier(tmp_path: Path) -> None:
    """A frozen case rendered back into the production briefing is a *reader*, and nothing wider:
    the facts tier is exactly the surface a re-run needs."""
    box = decide_input(_approved_case(tmp_path), context_window=WINDOW).toolbox

    assert isinstance(box, ResearchFacts)
    assert not isinstance(box, Toolbox)
