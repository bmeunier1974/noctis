"""The gate ladder in prose is held to the ladder in code (story #164, epic #158).

A promotion gate is only half-shipped while the pages that enumerate the ladder still tell the
old story: an operator reads ``docs/validation.md`` instead of ``promotion.py``, and a ladder
that is missing a gate — or lists one in the wrong place — is a confidently wrong sentence about
the arbiter of quality.

Three checks keep the pages honest, and each fails on a *code* change rather than on a human
remembering:

* every name in :data:`GATE_ORDER` / :data:`FINAL_GATES` has declared prose here, so a gate added
  to the module without a decision about how prose names it lands as a red test;
* every enumerating block runs in ``GATE_ORDER`` order and skips nothing inside the span it
  covers, so a gate inserted mid-ladder cannot quietly append itself to the end of a page;
* the pages that make a claim about *all* the gates ("these gates apply unchanged", "no gate is
  loosened") name the newest gate, which is exactly the claim a new gate invalidates.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from noctis.champions.promotion import FINAL_GATES, GATE_ORDER

ROOT = Path(__file__).resolve().parents[1]

# How each gate is allowed to be named in prose. Keyed by the gate name in code, so a gate the
# module grows tomorrow fails the first test until someone decides what to call it in English.
GATE_PROSE: dict[str, str] = {
    "validated": r"\bvalidated\b",
    "activity_floor": r"activity[ _-]floor",
    "overfit_gap": r"overfit[ _-]gap|gap[ _-]guard",
    "reverse_gap": r"reverse[ _-]gap",
    "magnitude_cap": r"magnitude[ _-]cap",
    "forward_holdout": r"forward[ _-](?:temporal[ _-])?holdout|forward[ _-]temporal[ _-]holdout",
    "symbol_holdout": r"symbol[ _-]holdout",
    "symbol_consistency": r"symbol[ _-]consistency|consistency",
    "family_slot": r"family[ _-]slot|one[ _-]slot[ _-]per[ _-]family|slot[ _-]per[ _-]family",
    "minimum_bar": r"minimum[ _-]bar|free[ _-]slot",
    "beat_weakest": r"beat[ _-](?:the[ _-])?weakest",
}

# The two degeneracy backstops are documented with the settings that switch them on
# (docs/configuration.md's promotion table) rather than in the prose ladders, which enumerate the
# gates an operator reasons about. They are exempt from the no-skipped-gate rule — nothing else
# is, so a new gate is never exempt by default.
SETTINGS_TABLE_ONLY = frozenset({"reverse_gap", "magnitude_cap"})

# Each block of prose that enumerates the ladder: (path, first line of the block, first line after
# it). Bounded rather than whole-page, because a page names a holdout long before it enumerates
# the gate that reads it.
LADDER_BLOCKS: tuple[tuple[str, str, str], ...] = (
    (
        "docs/validation.md",
        "**Quality gates** — each rejects outright on failure:",
        "The election metric itself",
    ),
    ("docs/research.md", "The gate order:", "Comparison is on a scale-free footing"),
    ("AGENTS.md", "2. **A failing strategy is a signal", "3. **No lookahead"),
    ("docs/run-record.md", "| `gate` | string |", "| `passed` |"),
)

# Pages whose claim is about *every* gate — "the gates apply to exit-bearing candidates
# unchanged", "no gate is loosened" — which is precisely the sentence a newly appended gate makes
# stale. Located by the paragraph that states it.
WHOLE_LADDER_CLAIMS: tuple[tuple[str, str], ...] = (
    ("docs/architecture.md", "Gate interaction:"),
    ("docs/plans/protective-exits-plan.md", "Gate interaction check"),
)


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _block(relative: str, start: str, end: str) -> str:
    text = _read(relative)
    first = text.find(start)
    assert first >= 0, f"{relative}: block marker {start!r} is gone — retarget this test"
    last = text.find(end, first + len(start))
    assert last >= 0, f"{relative}: end marker {end!r} is gone — retarget this test"
    return text[first:last]


def _paragraph(relative: str, marker: str) -> str:
    text = _read(relative)
    position = text.find(marker)
    assert position >= 0, f"{relative}: {marker!r} is gone — retarget this test"
    start = text.rfind("\n\n", 0, position)
    end = text.find("\n\n", position)
    return text[start if start >= 0 else 0 : end if end >= 0 else len(text)]


def _named_gates(block: str) -> list[str]:
    """The ladder gates the block names, in the order it first names each one."""
    found = [
        (match.start(), gate)
        for gate, pattern in GATE_PROSE.items()
        if (match := re.search(pattern, block, re.IGNORECASE))
    ]
    return [gate for _, gate in sorted(found)]


def test_every_gate_in_code_has_declared_prose() -> None:
    """A gate the module grows must be given an English name here before it can be swept for."""
    assert set(GATE_PROSE) == set(GATE_ORDER) | set(FINAL_GATES)


@pytest.mark.parametrize(("relative", "start", "end"), LADDER_BLOCKS)
def test_ladder_block_runs_in_gate_order(relative: str, start: str, end: str) -> None:
    """Every enumeration runs in the order the gates run, finals last."""
    order = [*GATE_ORDER, *FINAL_GATES]
    named = _named_gates(_block(relative, start, end))
    assert named == sorted(named, key=order.index), (
        f"{relative}: the gate ladder is enumerated out of order: {named}"
    )


@pytest.mark.parametrize(("relative", "start", "end"), LADDER_BLOCKS)
def test_ladder_block_skips_no_gate_inside_its_span(relative: str, start: str, end: str) -> None:
    """A page may enumerate part of the ladder, but never with a hole in the middle."""
    named = _named_gates(_block(relative, start, end))
    covered = [gate for gate in named if gate in GATE_ORDER]
    assert covered, f"{relative}: this block names no gate at all — retarget this test"
    span = GATE_ORDER[GATE_ORDER.index(covered[0]) : GATE_ORDER.index(covered[-1]) + 1]
    missing = [gate for gate in span if gate not in covered and gate not in SETTINGS_TABLE_ONLY]
    assert not missing, f"{relative}: the gate ladder skips {missing}"


@pytest.mark.parametrize(("relative", "marker"), WHOLE_LADDER_CLAIMS)
def test_whole_ladder_claim_accounts_for_the_newest_gate(relative: str, marker: str) -> None:
    """A claim about all the gates has to name the last one appended, or it is stale prose."""
    newest = GATE_ORDER[-1]
    paragraph = _paragraph(relative, marker)
    assert re.search(GATE_PROSE[newest], paragraph, re.IGNORECASE), (
        f"{relative}: the paragraph at {marker!r} claims something about every gate but never "
        f"names {newest!r}"
    )
