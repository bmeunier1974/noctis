"""The modules the operator pages name are held to the modules on disk (story #273, epic #264).

A refactor that moves a phase is only half-shipped while the pages still point at the old file:
a reader who follows ``src/noctis/engine/trading_day.py`` out of ``docs/architecture.md`` lands
on nothing, and prose that names a module the engine deleted is a confidently wrong sentence
about where the behaviour lives. That is exactly what this epic did — one trading module now,
three phase objects, one CLOSE order — so the pages get a check that fires on a **code** change
rather than on a human remembering:

* every module path a page spells out (``src/noctis/....py``, or a dotted ``noctis....``
  reference) resolves to something on disk, so deleting or renaming a module reddens the pages
  that still name it;
* each phase class the runtime holds is named in the two pages an agent reads first
  (``AGENTS.md``, ``docs/architecture.md``) *together with its own module file*, derived from
  the class object — rename ``ClosePhase`` or move ``close.py`` and this goes red;
* the glossary's **Close phase** entry names the close module, so a reader of the term lands on
  the code that implements it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from noctis.engine.close import ClosePhase
from noctis.engine.research_phase import ResearchPhase
from noctis.engine.trading_phase import TradingPhase
from noctis.observability.changelog import newest_entry

ROOT = Path(__file__).resolve().parents[1]

# The pages an operator or an agent reads to find out where behaviour lives.
PAGES: tuple[str, ...] = ("AGENTS.md", "CONTEXT.md", "README.md") + tuple(
    sorted(str(path.relative_to(ROOT)) for path in (ROOT / "docs").rglob("*.md"))
)

# `src/noctis/engine/close.py` — a spelled-out repository path.
MODULE_PATH = re.compile(r"`(src/noctis/[A-Za-z0-9_/]+\.py)`")
# `noctis.engine.close`, `noctis.bootstrap.build_families` — a dotted import path, which may
# name a module or a symbol inside one.
DOTTED_PATH = re.compile(r"`(noctis(?:\.[a-z0-9_]+)+)`")

# The phase objects the runtime holds, one per phase.
PHASE_CLASSES = (ResearchPhase, TradingPhase, ClosePhase)

# The two pages that are history rather than a map. A dated entry records what a change *did* —
# and a change may have deleted a module (#342 deletes `src/noctis/backtest/candidate.py`), so
# naming it there is the truth, not a dangling pointer. What still has to resolve on those pages
# is their living prose: the header and the component/site table a declaration is written against.
CHANGELOGS = ("docs/engine-changelog.md", "docs/prompt-changelog.md")


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _living_prose(page: str, text: str) -> str:
    """The half of a page that points at today's engine: all of it, or — for a changelog — the
    prose above its newest dated entry, found by the reader the ratchets themselves parse with."""
    if page not in CHANGELOGS:
        return text
    entry = newest_entry(text)
    return text if entry is None else text.split(f"## {entry.heading}", 1)[0]


def _resolves(dotted: str) -> bool:
    """True when ``dotted`` — or any prefix of it — is a module or package under ``src/``."""
    parts = dotted.split(".")
    while parts:
        base = ROOT / "src" / Path(*parts)
        if base.with_suffix(".py").is_file() or base.is_dir():
            return True
        parts.pop()
    return False


def _glossary_entry(title: str) -> str:
    text = _read("CONTEXT.md")
    start = text.find(f"## {title}\n")
    assert start >= 0, f"CONTEXT.md: the '{title}' entry is gone — retarget this test"
    end = text.find("\n## ", start + 1)
    return text[start : end if end >= 0 else len(text)]


@pytest.mark.parametrize("page", PAGES)
def test_every_module_path_a_page_spells_out_exists(page: str) -> None:
    """A page may not point at a module the engine no longer has."""
    prose = _living_prose(page, _read(page))
    missing = [path for path in MODULE_PATH.findall(prose) if not (ROOT / path).is_file()]
    assert not missing, f"{page} names module paths that do not exist: {missing}"


@pytest.mark.parametrize("page", PAGES)
def test_every_dotted_module_reference_resolves(page: str) -> None:
    """`noctis.engine.close` and friends resolve; a symbol resolves through its module."""
    prose = _living_prose(page, _read(page))
    missing = [dotted for dotted in DOTTED_PATH.findall(prose) if not _resolves(dotted)]
    assert not missing, f"{page} names dotted modules that do not exist: {missing}"


def test_a_changelogs_dated_entries_are_history_and_its_header_is_still_a_map() -> None:
    """The one page kind whose prose may name a deleted module: an entry declaring a deletion has
    to spell the file it deleted, while everything above the newest entry — the header and the
    component table the ratchet's declaration grammar is written against — still has to resolve."""
    text = (
        "# Engine changelog\n\n| `backtest` | `src/noctis/backtest/pipeline.py` |\n\n"
        "## 2026-08-25 — components: backtest — behaviour: unchanged\n\n"
        "`src/noctis/backtest/candidate.py` is deleted.\n"
    )

    assert MODULE_PATH.findall(_living_prose(CHANGELOGS[0], text)) == [
        "src/noctis/backtest/pipeline.py"
    ]
    assert MODULE_PATH.findall(_living_prose("docs/architecture.md", text)) == [
        "src/noctis/backtest/pipeline.py",
        "src/noctis/backtest/candidate.py",
    ]


@pytest.mark.parametrize("phase", PHASE_CLASSES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("page", ("AGENTS.md", "docs/architecture.md"))
def test_the_pages_name_each_phase_object_beside_its_own_module(page: str, phase: type) -> None:
    """Each phase class is named where a reader looks, together with the file it lives in."""
    text = _read(page)
    module_file = f"{phase.__module__.rsplit('.', 1)[-1]}.py"
    assert phase.__name__ in text, f"{page} never names {phase.__name__}"
    assert module_file in text, f"{page} names {phase.__name__} but not its module {module_file}"


def test_the_glossary_close_phase_entry_lands_on_the_close_module() -> None:
    """A reader of the term reaches the code: the entry names ClosePhase and its module."""
    entry = _glossary_entry("Close phase")
    assert ClosePhase.__name__ in entry
    assert ClosePhase.__module__ in entry or "close.py" in entry
