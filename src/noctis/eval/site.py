"""``AgentSite`` — one frozen declaration naming everything that defines one LLM judgment site.

Each of the engine's benchmarkable call sites is already complete and correct, but scattered: the
typed emit contract lives with the episodic driver, the briefing builder in the briefings module,
the knobs in the research config, the retry policy in the episode runner. Nothing is wrong — but
nothing says "this is the formulate site, here is everything that defines it", so a benchmark
harness has nothing to look up. This type is that sentence, and nothing more:

* ``id`` — the site's name (``coder``, ``formulate``, ``decide``, ``discover``, ``distill``).
* ``version`` — the contract *generation*, bumped by hand when the site's emit contract (or, for
  the coder, its brief/gate interface) changes shape, so a record can refuse a comparison across
  generations. The same declared-version discipline the engine version integer uses.
* ``contract`` — the site's :class:`~noctis.research.episode.EmitContract`, **the object the driver
  itself emits through**, or ``None`` for a site whose output is free-form and judged by a gate
  instead (the coder: one fenced code block, judged by the fresh-subprocess write gate). The
  declaration model represents that honestly rather than inventing a contract that does not exist.
* ``render`` — input plus a :class:`~noctis.eval.harness.HarnessSpec` in, one prompt string out.
  A declaration binds the production builder, through a thin forwarding adapter where the builder's
  keyword signature differs; an adapter may never re-order, filter or rewrite what the builder
  produces, because then production and benchmark would stop exercising the same code.
* ``knobs`` — the :class:`~noctis.eval.knobs.SiteKnobs` subclass this site accepts, so a harness
  can refuse a bad override before spending money.
* ``scorers`` — per-site, never shared: the site's own reading of a finished bench (see
  :class:`Scorer`). Empty is the honest default — a site whose answers only the emit contract judges
  publishes the eval core's figures and adds nothing of its own.
* ``difficulty_axes`` — the axis vocabulary this site's corpus labels its cases on, in the order a
  reading publishes them. A benchmark splits its headline figure by these (an agreement figure that
  is one thing on near-margin cases and another on comfortable ones is two findings, not one), and
  the runner hands the tuple to the site's scorers so one site-neutral loop
  (:func:`~noctis.eval.reading.strata`) can stratify either site's reading without importing
  either's vocabulary. Empty is the honest default: cases nobody labelled have no axes, which is an
  absence and not an empty measurement.

**Declaration-only, and frozen.** Constructing an ``AgentSite`` runs nothing and changes nothing;
a site is data a harness reads. That is also what keeps the door open for a future DSPy-style
prompt optimizer: sites as data need no further engine change to become an optimizer's entry point.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from noctis.eval.case import Case
from noctis.eval.harness import HarnessSpec
from noctis.eval.knobs import SiteKnobs
from noctis.eval.reading import SiteReading
from noctis.research.episode import EmitContract

__all__ = ["AgentSite", "AnsweredCase", "Scorer", "SiteInput", "SiteOutput"]

# The two halves of a site's shape: what it is asked (a briefing, a brief, a findings history) and
# what it answers with (the typed record its contract parses, or source text for the coder).
SiteInput = TypeVar("SiteInput")
SiteOutput = TypeVar("SiteOutput")

# The scoring protocol's own parameters. They stay covariant because neither appears in its one
# member: a scoring pass reads *answers*, which are text a contract has yet to be applied to, so it
# constrains neither half of the site's shape.
ScoredInput_co = TypeVar("ScoredInput_co", covariant=True)
ScoredOutput_co = TypeVar("ScoredOutput_co", covariant=True)


@dataclass(frozen=True)
class AnsweredCase:
    """One finished bench job as a scoring pass sees it: the case asked, and what came back.

    ``replies`` is one entry per attempt, in the order the runner made them, carrying the answer
    text it retained (``None`` for an attempt that produced none — a provider that fell over). The
    retry loop stops at the first pass, so :attr:`settled` — the last of them — is the answer this
    job actually ended on, and it is the one a scorer grades.

    Deliberately plain values rather than the runner's own types: a pooled bench returns its work
    across a pickle boundary, and a scoring surface that could not cross one would quietly make
    ``--workers`` mean something different from ``--workers 1``.
    """

    case: Case
    config_id: str | None = None
    rep: int = 1
    replies: tuple[str | None, ...] = ()

    @property
    def settled(self) -> str | None:
        """The reply this job ended on, or ``None`` when it never produced one."""
        return self.replies[-1] if self.replies else None


class Scorer(Protocol[ScoredInput_co, ScoredOutput_co]):
    """One site's own reading of a finished bench — the numbers only that site knows how to compute.

    The eval core scores what every site shares: whether the ask emitted at all, in how many
    attempts, at what cost. A *site's* scorer is the other half — DECIDE's approval-side agreement
    against the labels the promotion gates recorded — and it is per-site, never shared, because that
    arithmetic is meaningless anywhere else.

    **The one member, and why it is stated here (#213).** This slot shipped memberless as a forward
    declaration: guessing a scoring signature before a real scorer existed would have been a
    decision made in the wrong epic. There is now one real scorer and one caller, so the protocol
    states exactly what that caller needs and nothing more — every answered case in, the reading a
    record quotes under ``harness.dials`` out, or ``None`` from a scorer with nothing to say. The
    runner folds whatever comes back into the dials without interpreting it, which is what keeps the
    bench verbs free of any site's vocabulary.

    **What comes back is a type, not a mapping (#308).** A reading is a
    :class:`~noctis.eval.reading.SiteReading`: the headline facts a record states above the block,
    the co-primary pair inside it, the counts, rows, per-axis strata and the site's own sections —
    and one :meth:`~noctis.eval.reading.SiteReading.as_dials` that publishes them. The shape every
    published reading shares used to live only in the literal each site assembled by hand and in
    the golden that pinned it; it is declared once now, and the live sites and the retrospective
    miner build the same object. The import runs downward on purpose — this module reads
    :mod:`noctis.eval.reading`, which is stdlib-only and knows no site.
    """

    def read(
        self, answered: Sequence[AnsweredCase], *, axes: tuple[str, ...] = ()
    ) -> SiteReading | None:
        """This site's reading over one bench's answers, or ``None`` when it has none to add.

        ``axes`` is the site's declared :attr:`AgentSite.difficulty_axes`, handed over by the runner
        rather than looked up: a scorer is a stateless singleton its declaration carries, so it
        cannot import the declaration that carries it, and the declaration stays the one place a
        site's facts live. A pass handed no axes stratifies by none.
        """


@dataclass(frozen=True)
class AgentSite(Generic[SiteInput, SiteOutput]):
    """One benchmarkable LLM call site, declared as data — a field-by-field tour is above."""

    id: str
    version: str
    contract: EmitContract[SiteOutput] | None
    render: Callable[[SiteInput, HarnessSpec], str]
    knobs: type[SiteKnobs]
    scorers: tuple[Scorer[SiteInput, SiteOutput], ...] = ()
    difficulty_axes: tuple[str, ...] = ()
