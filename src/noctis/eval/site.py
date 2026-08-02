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
* ``scorers`` — per-site, never shared. Empty until the eval core lands (see :class:`Scorer`).

**Declaration-only, and frozen.** Constructing an ``AgentSite`` runs nothing and changes nothing;
a site is data a harness reads. That is also what keeps the door open for a future DSPy-style
prompt optimizer: sites as data need no further engine change to become an optimizer's entry point.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from noctis.eval.harness import HarnessSpec
from noctis.eval.knobs import SiteKnobs
from noctis.research.episode import EmitContract

__all__ = ["AgentSite", "Scorer", "SiteInput", "SiteOutput"]

# The two halves of a site's shape: what it is asked (a briefing, a brief, a findings history) and
# what it answers with (the typed record its contract parses, or source text for the coder).
SiteInput = TypeVar("SiteInput")
SiteOutput = TypeVar("SiteOutput")

# The placeholder protocol's own parameters. They are covariant because a protocol with no members
# places no constraint on either half; the real variance is a decision for whoever states the
# scoring signature (plan 03), which is exactly the decision this story declines to make.
ScoredInput_co = TypeVar("ScoredInput_co", covariant=True)
ScoredOutput_co = TypeVar("ScoredOutput_co", covariant=True)


class Scorer(Protocol[ScoredInput_co, ScoredOutput_co]):
    """Forward declaration only — the generic eval core (plan 03) owns and will replace this.

    It exists so a site's ``scorers`` slot can be *typed* today while shipping empty: the first
    real scorers arrive with the DECIDE and coder epics, and their protocol is theirs to state.
    Deliberately memberless: guessing a scoring signature here would be a decision made in the
    wrong epic.
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
