"""The bench reading — one ``bench.json``, rendered for a human. Pure: a record in, text out.

This is the reader half of the bench record, and it is deliberately **generic**. It knows the
record's own shape (a bench block, a metrics block, a verbatim ``harness.dials`` subtree) and
nothing whatsoever about the site that produced it: there is no branch on ``site_id`` anywhere
below, no DECIDE vocabulary, and no list of expected axes. A record whose dials publish an approval
pair and three difficulty axes renders those; a record from a site with one figure and one axis
renders that; a record with no dials at all still renders its metrics. **The reading appears because
the record carries it**, which is what keeps one verb honest across every site the eval layer grows.

Three rules do all the work:

* **The co-primary pair is rendered by the pairing rule, or not at all.** A dials block carrying the
  whole :class:`~noctis.eval.decide_scorer.ApprovalPair` is handed to that type and printed by
  :meth:`~noctis.eval.decide_scorer.ApprovalPair.render`, which cannot print agreement without the
  approval rate beside it (#206). A block carrying an ``agreement`` figure *without* the rest of the
  pair is **refused** — the figure is named and not printed — because a bare agreement is exactly
  the half-truth the pairing rule exists to prevent, and re-implementing the rendering here to be
  helpful would be re-implementing the loophole.
* **An absence renders as the literal ``n/a``, never as a zero.** A measured zero is a finding and
  prints as ``0.0000``; a ``null`` is a missing input and prints as ``n/a``. The distinction is the
  record's, and this module's only job is not to lose it.
* **Nesting is the breakdown.** Every nested mapping renders as its own indented block under the
  record's own key, so ``strata → axis → level → figures`` needs no special case: the per-axis
  breakdown is the recursion, and an axis nobody has invented yet renders the day a record carries
  it.

Nothing here reads a file, a clock or a configuration — the verb (``bench report``) resolves and
validates the document, and this module renders whatever it is handed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Any

from noctis.eval.decide_scorer import ApprovalPair

__all__ = ["NOT_AVAILABLE", "render_bench_report"]

#: How every unknown figure renders, at every depth. One token, spelled once.
NOT_AVAILABLE = "n/a"

# The two-column table's widths — the pairing rule's own (``decide_scorer._table``), so a block
# printed by :meth:`ApprovalPair.render` lines up with the rows this module prints beside it.
_LABEL_WIDTH = 28
_VALUE_WIDTH = 12

# One nesting level, in spaces. A whole block is shifted by it, values included, so a stratum's
# figures stay column-aligned with the pair's own rendering inside the same block.
_INDENT = "  "

# A defensive ceiling on recursion: a record is data from disk, and a reader that could be made to
# recurse forever by a hand-edited document is a reader with a crash in it.
_MAX_DEPTH = 8

# The fields a block must carry before it may be rendered as the co-primary pair. Read off the type
# rather than restated, so a pair that grows a field cannot be half-rendered from here.
_PAIR_FIELDS: tuple[str, ...] = tuple(field.name for field in fields(ApprovalPair))

# The one figure that may never be printed on its own. Named here because the refusal has to know
# what it is refusing; the *rendering* of it lives entirely in :class:`ApprovalPair`.
_UNPAIRED = "agreement"

_PAIR_REFUSAL = (
    f"REFUSED — this block carries {_UNPAIRED!r} without the approval rate and counts that "
    "qualify it. Agreement alone is trivially gamed (approve one candidate, score 1.0), so the "
    "co-primary pair renders whole or not at all."
)


def render_bench_report(record: Mapping[str, Any], *, source: object = None) -> str:
    """One bench record as the report ``bench report`` prints. Pure, and total over any document.

    ``source`` is the file the record was read from, named in the header when it is given — a
    report an operator can trace back to bytes on disk. Every section is tolerant of a record that
    does not carry it: a missing block is an empty one, and every unknown figure is ``n/a``.
    """
    sections = [
        _header(record, source),
        _population(record),
        # The record's own reading first — it is the headline, and it is the part that differs
        # site by site — then the metrics block every bench record carries in the same shape.
        *_dials(record),
        _titled("Metrics", _section(record, "metrics")),
    ]
    return "\n\n".join(section for section in sections if section)


# ── the header: what this bench was, and what its numbers are comparable under ───────────────


def _header(record: Mapping[str, Any], source: object) -> str:
    """The identity block: the bench, the site, the split, and the key that buckets the numbers."""
    bench = _section(record, "bench")
    site = _section(record, "site")
    corpus = _section(record, "corpus")
    label = bench.get("label")
    lines = [f"Bench {_text(bench.get('bench_id'))}" + (f" — {label}" if label else "")]
    if source is not None:
        lines.append(f"{_INDENT}{source}")
    lines += [
        "",
        _line("Site", site.get("site_id")),
        _line("Site version", site.get("site_version")),
        _line("Corpus split", corpus.get("split")),
        _line("Complete", bench.get("complete")),
        _line("Started (UTC)", bench.get("started_utc")),
        _line("Finished (UTC)", bench.get("finished_utc")),
        "Comparable key",
        f"{_INDENT}{_text(record.get('comparable_key'))}",
    ]
    return "\n".join(lines)


def _population(record: Mapping[str, Any]) -> str:
    """How much was measured, and how much of the corpus was not — n and reps, named.

    ``Reps per case`` is ``n/a`` unless the runs divide evenly across the cases: an uneven bench has
    no single rep count, and rounding one into existence would misdescribe the population every rate
    above is a rate over.
    """
    bench = _section(record, "bench")
    corpus = _section(record, "corpus")
    cases, runs = _count(bench.get("cases")), _count(bench.get("runs"))
    corpus_cases = _count(corpus.get("case_count"))
    reps = runs // cases if cases and runs is not None and runs % cases == 0 else None
    unscored = None if corpus_cases is None or cases is None else corpus_cases - cases
    return "\n".join(
        [
            _line("Cases (n)", cases),
            _line("Runs (reps)", runs),
            _line("Reps per case", reps),
            _line("Attempts", bench.get("attempts")),
            _line("Corpus cases", corpus_cases),
            _line(f"{_INDENT}not scored", unscored),
        ]
    )


# ── the generic blocks: metrics, dials, and every reading a record carries ───────────────────


def _dials(record: Mapping[str, Any]) -> list[str]:
    """The verbatim harness subtree, as one scalar block plus one section per reading it carries.

    The record quotes its dials rather than authoring them, and so does this: a scalar dial is a
    row, and a dial that is itself a mapping is a section named by the record's own key. That is
    where a site's reading — its figures, its strata, its axes — arrives, and nothing here needs to
    know which site wrote it.
    """
    harness = _section(record, "harness")
    dials = harness.get("dials")
    if not isinstance(dials, Mapping):
        return [f"Harness dials\n{_INDENT}{NOT_AVAILABLE} — this record carries no dials"]
    scalars = {key: value for key, value in dials.items() if not isinstance(value, Mapping)}
    sections = [_titled("Harness dials", scalars)] if scalars else []
    return sections + [
        _titled(f"dials.{key}", value) for key, value in dials.items() if isinstance(value, Mapping)
    ]


def _titled(title: str, node: Mapping[str, Any] | None) -> str:
    """One titled block, or nothing at all when the record carries no such section."""
    if not isinstance(node, Mapping):
        return ""
    body = _block(node, depth=0)
    return "\n".join([title, *body]) if body else f"{title}\n{_INDENT}(empty)"


def _block(node: Mapping[str, Any], *, depth: int) -> list[str]:
    """One mapping, rendered: the pair first if it is one, then its rows and its sub-blocks.

    The recursion *is* the per-axis breakdown — a mapping under a key becomes an indented block
    headed by that key, however deep the record nests it — so nothing here enumerates axes, levels
    or figure names.
    """
    if depth > _MAX_DEPTH:
        return [f"{_INDENT * depth}(nested deeper than this report renders)"]
    lines, rendered = _pair_lines(node, depth=depth)
    for key, value in node.items():
        if key in rendered:
            continue
        name = str(key)
        if isinstance(value, Mapping):
            lines.append(f"{_INDENT * depth}{name}")
            nested = _block(value, depth=depth + 1)
            lines += nested if nested else [f"{_INDENT * (depth + 1)}(empty)"]
        elif _is_list(value):
            lines.append(_line(name, f"[{len(value)} entries]", depth=depth))
        else:
            lines.append(_line(name, value, depth=depth))
    return lines


def _pair_lines(node: Mapping[str, Any], *, depth: int) -> tuple[list[str], set[str]]:
    """The co-primary pair of this block — whole, refused, or absent — and the keys it consumed.

    Whole: the pairing rule's own renderer prints it, agreement and approval rate together. Refused:
    the block carries an agreement without the rest of the pair, so the figure is named and *not*
    printed. Absent: nothing here is a pair, and every key falls through to the generic rows.
    """
    if not set(_PAIR_FIELDS) <= set(node):
        if _UNPAIRED in node:
            return [f"{_INDENT * depth}{_PAIR_REFUSAL}"], {_UNPAIRED}
        return [], set()
    pair = ApprovalPair(**{name: node[name] for name in _PAIR_FIELDS})
    return _shift(pair.render(), depth=depth), set(_PAIR_FIELDS)


# ── formatting: one place where an absence becomes the word ``n/a`` ──────────────────────────


def _line(label: str, value: object, *, depth: int = 0) -> str:
    """One label/value row, indented whole so a nested block stays column-aligned."""
    return f"{_INDENT * depth}{label:<{_LABEL_WIDTH}}{_fmt(value):>{_VALUE_WIDTH}}"


def _fmt(value: object) -> str:
    """A cell: ``n/a`` for a null, four decimals for a rate, yes/no for a flag, the value else."""
    if value is None:
        return NOT_AVAILABLE
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _text(value: object) -> str:
    """One free-text field — ``n/a`` when the record does not carry it."""
    return NOT_AVAILABLE if value is None else str(value)


def _shift(rendered: str, *, depth: int) -> list[str]:
    """Another module's rendered block, indented into this one — never re-rendered here."""
    return [f"{_INDENT * depth}{line}" for line in rendered.splitlines()]


def _section(record: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """One section of a record, or an empty one: a reader is tolerant of what it is handed."""
    value = record.get(key) if isinstance(record, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _count(value: object) -> int | None:
    """One counted field as a number, or ``None`` for anything that is not one."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _is_list(value: object) -> bool:
    """Whether a value is a list of things rather than a scalar (a string is a scalar)."""
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)
