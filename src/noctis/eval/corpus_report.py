"""The corpus reading — what a site's cases are, how they are labelled, how they are divided.

The twin of :mod:`noctis.eval.bench_report`, one artifact earlier: that module reads a *bench
record* (what a run over a corpus produced), this one reads the **corpus itself** (what there is to
run over). Both are pure — data in, text out, no directory and no clock — because the thing that
reads a corpus must be cheap enough to test without one, and because a report that could touch its
subject is a report an operator has to think about before running.

**Everything it counts has already been validated.** This module never loads a file: the caller
hands it the cases a site's own provider admitted, so a corpus that would not load is a refusal at
the verb rather than a number computed over the files that happened to parse.

**The vocabulary is declared, so an absence can be named.** A count over what is *present* can only
say what somebody wrote; :class:`CorpusVocabulary` carries what the site *declares* — its buckets,
and each difficulty axis's closed value set — so a bucket nobody has filed a case in and an axis
level nobody has written a case for are printed as declared-and-unrepresented rather than silently
missing from the table. A site that declares none of it (the generic flat corpora) is reported over
exactly what its cases carry, which is the honest reading when there is no vocabulary to compare
against. Nothing here names a site.

**Three counts per group, and the third is the interesting one.** ``tuning`` and ``holdout`` are the
split as the corpus deals it; ``unstamped`` is how many of those cases got their half from that deal
rather than from their own file. The distinction matters because the deal is deterministic but not
*frozen* — an unstamped case can move when the corpus grows (see :mod:`noctis.eval.corpus`) — so a
curator reading ``unstamped 3`` is reading "three cases still need
:func:`~noctis.eval.coder_corpus.stamp_splits`", which no other view of the corpus states.

**Refusal-first rendering**, the house style ``bench report`` set: an absent figure is the literal
``n/a`` and never a zero, because a share over an empty stratum is not a share and a zero there
would read as a finding.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from noctis.eval.bench_report import NOT_AVAILABLE
from noctis.eval.case import Case, Split
from noctis.eval.corpus import Corpus

__all__ = [
    "AxisReading",
    "CorpusReading",
    "CorpusVocabulary",
    "SplitBalance",
    "Stratum",
    "read_corpus",
    "render_corpus_report",
]

# One nesting level, in spaces — the bench report's own, so the two verbs' output reads as one tool.
_INDENT = "  "

# How wide a group's name is padded before its counts. Wide enough for the longest declared axis
# level a site ships (``higher_timeframe``, ``param_space_breadth``) at the depth it prints at.
_NAME_WIDTH = 24

# What a group with no cases in it says instead of shares that do not exist.
_UNREPRESENTED = f"{NOT_AVAILABLE} (declared, no case represents it)"
_EMPTY = f"{NOT_AVAILABLE} (no cases)"


@dataclass(frozen=True)
class SplitBalance:
    """How one group of cases divides, and how much of that division is still unfrozen.

    ``tuning`` and ``holdout`` are the halves as the corpus deals them, so they always account for
    every case in the group; ``unstamped`` counts the subset whose *file* declared no ``split:`` and
    that therefore took its half from the deal — a number that overlaps both halves on purpose.
    """

    tuning: int = 0
    holdout: int = 0
    unstamped: int = 0

    @property
    def cases(self) -> int:
        """How many cases this group holds — every one of them is in exactly one half."""
        return self.tuning + self.holdout

    @property
    def tuning_share(self) -> float | None:
        """The tuning half's share of this group, or ``None`` when the group is empty."""
        return None if not self.cases else self.tuning / self.cases

    @property
    def holdout_share(self) -> float | None:
        """The holdout half's share of this group, or ``None`` when the group is empty."""
        return None if not self.cases else self.holdout / self.cases


@dataclass(frozen=True)
class Stratum:
    """One group a corpus is counted over: a bucket, or one level of one difficulty axis.

    ``declared`` says the site's own vocabulary names this group, which is what lets an empty one be
    reported as *unrepresented* rather than merely absent.
    """

    name: str
    balance: SplitBalance
    declared: bool = False

    @property
    def represented(self) -> bool:
        """Whether any case in the corpus falls in this group."""
        return self.balance.cases > 0


@dataclass(frozen=True)
class AxisReading:
    """One difficulty axis and every level of it a reader should be told about, in order."""

    axis: str
    levels: tuple[Stratum, ...] = ()


@dataclass(frozen=True)
class CorpusVocabulary:
    """What a site declares about how its cases are labelled — the lookup's half of the reading.

    ``bucket_of`` is how a case's bucket is read (``None`` for a site that has no buckets at all,
    which is most of them); ``buckets`` and ``axis_levels`` are the closed vocabularies an absence
    is named against. All three are supplied by :data:`~noctis.eval.bootstrap.SITE_CORPORA`, so this
    module needs no branch on a site id and a new site's row is a reviewable diff there.
    """

    bucket_of: Callable[[Case], str] | None = None
    buckets: tuple[str, ...] = ()
    axis_levels: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class CorpusReading:
    """One corpus, counted: its identity, its split balance, and its stratification.

    ``buckets`` is ``None`` — not empty — for a site whose cases carry no buckets, because "this
    site has none" and "this site's buckets are all empty" are different facts and a reader must
    not have to guess which one an empty table meant.
    """

    site_id: str
    case_count: int
    digest: str
    overall: SplitBalance
    buckets: tuple[Stratum, ...] | None = None
    axes: tuple[AxisReading, ...] = ()


def read_corpus(
    site_id: str, cases: Sequence[Case], *, vocabulary: CorpusVocabulary | None = None
) -> CorpusReading:
    """Count one site's already-validated cases into the reading its report is rendered from.

    The cases are dealt into a :class:`~noctis.eval.corpus.Corpus` here — the same deal a bench
    measures under, so the balance reported is the balance that would be run — while the *files'*
    own assignments are read off the cases as they arrived, which is where ``unstamped`` comes from.
    A corpus that cannot be one (a repeated case id, a foreign site) refuses out of this call.
    """
    speaks = vocabulary if vocabulary is not None else CorpusVocabulary()
    unstamped = frozenset(case.case_id for case in cases if case.split is None)
    corpus = Corpus(site_id=site_id, cases=tuple(cases))
    dealt = corpus.cases
    return CorpusReading(
        site_id=site_id,
        case_count=len(corpus),
        digest=corpus.digest,
        overall=_balance(dealt, unstamped),
        buckets=_buckets(dealt, unstamped, speaks),
        axes=_axes(dealt, unstamped, speaks),
    )


def render_corpus_report(
    reading: CorpusReading, *, source: object = None, loader: str | None = None
) -> str:
    """One corpus reading as the report ``bench corpus`` prints. Pure, and total over any reading.

    ``source`` is the directory the cases were read from and ``loader`` the provider that read them
    — both named when they are given, because "20 cases validated" is only an answer beside "from
    where, and through what".
    """
    sections = [
        _header(reading, source, loader),
        _titled("Split balance", [_stratum_line(Stratum("corpus", reading.overall), depth=1)]),
        _titled("Buckets", _bucket_lines(reading)),
        _titled("Axes", _axis_lines(reading)),
    ]
    return "\n\n".join(section for section in sections if section)


# ── the sections ─────────────────────────────────────────────────────────────────────────────


def _header(reading: CorpusReading, source: object, loader: str | None) -> str:
    """What was read, from where, through what — and the hash of exactly these cases."""
    through = f" through {loader}" if loader else ""
    return "\n".join(
        [
            f"Corpus {reading.site_id}" + (f" — {source}" if source is not None else ""),
            f"{_INDENT}{reading.case_count} case(s) validated{through} — digest {reading.digest}",
        ]
    )


def _bucket_lines(reading: CorpusReading) -> list[str]:
    """One row per bucket, or the one honest line a site without buckets earns."""
    if reading.buckets is None:
        return [f"{_INDENT}{NOT_AVAILABLE} — the {reading.site_id} corpus declares no buckets"]
    return [_stratum_line(bucket, depth=1) for bucket in reading.buckets]


def _axis_lines(reading: CorpusReading) -> list[str]:
    """One block per axis, each headed by the axis name and holding one row per level."""
    if not reading.axes:
        return [f"{_INDENT}{NOT_AVAILABLE} — no case declares a difficulty axis"]
    lines: list[str] = []
    for axis in reading.axes:
        lines.append(f"{_INDENT}{axis.axis}")
        lines.extend(_stratum_line(level, depth=2) for level in axis.levels)
    return lines


def _titled(title: str, body: Sequence[str]) -> str:
    """One titled block, or nothing at all when there is no body to put under it."""
    return "\n".join([title, *body]) if body else ""


def _stratum_line(stratum: Stratum, *, depth: int) -> str:
    """One group's row: its name, how many cases it holds, and how those cases divide."""
    name = f"{_INDENT * depth}{stratum.name:<{_NAME_WIDTH}}"
    counted = f"{stratum.balance.cases:>3} case(s)"
    if not stratum.represented:
        return f"{name}{counted} — {_UNREPRESENTED if stratum.declared else _EMPTY}"
    balance = stratum.balance
    return (
        f"{name}{counted} — "
        f"tuning {balance.tuning} ({_share(balance.tuning_share)}), "
        f"holdout {balance.holdout} ({_share(balance.holdout_share)}), "
        f"unstamped {balance.unstamped}"
    )


def _share(value: float | None) -> str:
    """One share, to four decimals — ``n/a`` when there was nothing to take a share of."""
    return NOT_AVAILABLE if value is None else f"{value:.4f}"


# ── the counting ─────────────────────────────────────────────────────────────────────────────


def _balance(cases: Iterable[Case], unstamped: frozenset[str]) -> SplitBalance:
    """How one group of dealt cases divides, with the files' own unstamped ones counted."""
    members = tuple(cases)
    return SplitBalance(
        tuning=sum(1 for case in members if case.split is Split.TUNING),
        holdout=sum(1 for case in members if case.split is Split.HOLDOUT),
        unstamped=sum(1 for case in members if case.case_id in unstamped),
    )


def _buckets(
    cases: Sequence[Case], unstamped: frozenset[str], speaks: CorpusVocabulary
) -> tuple[Stratum, ...] | None:
    """One stratum per bucket — declared ones first, always, then anything else the cases carry."""
    if speaks.bucket_of is None:
        return None
    grouped = _grouped(cases, speaks.bucket_of)
    return _strata(grouped, speaks.buckets, unstamped)


def _axes(
    cases: Sequence[Case], unstamped: frozenset[str], speaks: CorpusVocabulary
) -> tuple[AxisReading, ...]:
    """One reading per axis, over the axes the site declares and the ones its cases carry."""
    carried = sorted({axis for case in cases for axis in case.difficulty})
    readings = []
    for axis in _ordered(speaks.axis_levels, carried):
        grouped: dict[str, list[Case]] = {}
        for case in cases:
            level = case.difficulty.get(axis)
            if level is not None:
                grouped.setdefault(level, []).append(case)
        levels = _strata(grouped, speaks.axis_levels.get(axis, ()), unstamped)
        readings.append(AxisReading(axis=axis, levels=levels))
    return tuple(readings)


def _grouped(cases: Sequence[Case], key: Callable[[Case], str]) -> dict[str, list[Case]]:
    """The cases of one corpus, gathered under the label a reader groups them by."""
    grouped: dict[str, list[Case]] = {}
    for case in cases:
        grouped.setdefault(key(case), []).append(case)
    return grouped


def _strata(
    grouped: Mapping[str, Sequence[Case]], declared: Sequence[str], unstamped: frozenset[str]
) -> tuple[Stratum, ...]:
    """Every group a reader is told about: the declared vocabulary in its own order, then the rest.

    A declared group with no cases is kept — that absence is the finding — and a group the cases
    carry that the vocabulary does not name is kept too, because dropping it would hide a case.
    """
    return tuple(
        Stratum(
            name=name,
            balance=_balance(grouped.get(name, ()), unstamped),
            declared=name in set(declared),
        )
        for name in _ordered({name: () for name in declared}, sorted(grouped))
    )


def _ordered(declared: Mapping[str, Any], carried: Sequence[str]) -> tuple[str, ...]:
    """The declared names in their declared order, then whatever else was carried, sorted."""
    return (*declared, *(name for name in carried if name not in declared))
