"""The three episodic judgment sites, declared: **formulate**, **decide**, **discover**.

Each declaration names production objects and nothing else. The ``contract`` field *is* the
:class:`~noctis.research.episode.EmitContract` the episodic driver
(:mod:`noctis.research.driver`) emits through — the same object, by identity, never a copy — and
each ``render`` forwards to the briefing builder in :mod:`noctis.research.briefings` that a real
session renders its prompt with. A benchmark that measured a copy would measure a fork, and a fork
drifts silently: the day someone edits the production prompt, the bench would keep scoring the old
one and report an improvement nobody shipped.

**The adapters are forwarding only.** ``AgentSite.render`` has one uniform shape,
``(input, HarnessSpec) -> str``; the builders have keyword signatures of their own (a toolbox, a
session ledger, a strategy name, a band profile, a context window). The gap is closed by a thin
closure per site plus a frozen input record naming exactly the arguments that builder takes — the
adapter re-orders nothing, filters nothing and rewrites nothing it gets back. It also does not yet
*read* the :class:`~noctis.eval.harness.HarnessSpec`: the composition ablations (contract sheet,
template, worked example, feasibility rules, retry hints) are wired into the builders by the
prompt-ablation epic, so today the spec is accepted, carried as part of the declared signature, and
otherwise unused. That is stated here rather than hidden behind a parameter that looks live.

**Versions are hand-bumped.** Every site starts at ``"1"`` and moves only when its emit contract
changes shape, so a benchmark record can refuse to compare results across generations. Decide is
the one site with two contracts — the primary ask and the revise-less final ask (#99) — and they
ride **one** version (see :data:`DECIDE_CONTRACTS`), because they are one judgment whose emit
vocabulary narrows: a change to either is a change to the decide site.

**The knob sets name settings knobs; they embed no Settings.** Each site accepts the same four
levers today — the session ``model``, the ``max_tokens`` output ceiling, the ``context_window``
that shapes the briefing's budget, and ``episode_retries``, the misfire retry count — every one an
existing field of ``research.agent`` (:class:`~noctis.config.settings.AgentResearchConfig`). They
are declared per site rather than shared so one site's levers can diverge without silently widening
another's, and each field defaults to ``None`` meaning *unset — whatever the session resolves*: a
knob set records what a bench may override, never a second copy of production's defaults.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from noctis.eval.decide_site import DECIDE_SCORER, DIFFICULTY_AXES
from noctis.eval.harness import HarnessSpec
from noctis.eval.knobs import SiteKnobs
from noctis.eval.site import AgentSite
from noctis.research.briefings import decide_briefing, discover_briefing, formulate_briefing
from noctis.research.driver import (
    DECIDE_CONTRACT,
    DECIDE_FINAL_CONTRACT,
    DISCOVER_CONTRACT,
    FORMULATE_CONTRACT,
    DecideOutput,
    DiscoverOutput,
    FormulateOutput,
)
from noctis.research.episode import EmitContract
from noctis.research.ledger import SessionLedger
from noctis.research.mandate import Mandate
from noctis.research.surface import ResearchFacts

__all__ = [
    "DECIDE_CONTRACTS",
    "DECIDE_SITE",
    "DISCOVER_SITE",
    "EPISODIC_SITES",
    "FORMULATE_SITE",
    "DecideKnobs",
    "DecideSiteInput",
    "DiscoverKnobs",
    "DiscoverSiteInput",
    "EpisodeKnobs",
    "FormulateKnobs",
    "FormulateSiteInput",
]

# The generation every episodic declaration starts at. Bumped by hand — per site — when that
# site's emit contract changes shape; never derived from a file hash, because a version a machine
# moves is a version nobody has to justify.
_INITIAL_VERSION = "1"


# ─────────────────────────────────────────────────────────────────────────────
# Knob sets — names of existing research.agent settings, one type per site
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EpisodeKnobs(SiteKnobs):
    """The levers every episodic site shares, named exactly as ``research.agent`` names them.

    ``None`` is *not* a default value for the knob — it means the bench overrides nothing and the
    session's own configuration stands. Restating production's defaults here would be a second
    source for a number that already has one.
    """

    model: str | None = None
    max_tokens: int | None = None
    context_window: int | None = None
    episode_retries: int | None = None


@dataclass(frozen=True)
class FormulateKnobs(EpisodeKnobs):
    """FORMULATE's levers. Its own type so the thesis ask can gain a knob decide never sees."""


@dataclass(frozen=True)
class DecideKnobs(EpisodeKnobs):
    """DECIDE's levers, covering both the primary and the final (revise-less) ask."""


@dataclass(frozen=True)
class DiscoverKnobs(EpisodeKnobs):
    """DISCOVER's levers — the smallest ask of the three, on the same four dials today."""


# ─────────────────────────────────────────────────────────────────────────────
# Site inputs — one frozen record per builder signature
# ─────────────────────────────────────────────────────────────────────────────
# The toolbox is typed as it is at the builders' own boundary: a
# :class:`~noctis.research.surface.ResearchFacts` — the read-only tier of the research-toolbox
# surface (#257). The briefing builders ask a session for *facts*, so that is exactly what a site
# input carries, and a session's real ResearchToolbox, a bench's stated context and the frozen
# case's stand-in all satisfy it structurally. A renderer is handed no tool it could dispatch.
@dataclass(frozen=True)
class FormulateSiteInput:
    """What :func:`~noctis.research.briefings.formulate_briefing` is called with."""

    toolbox: ResearchFacts
    ledger: SessionLedger
    context_window: int
    mandate: Mandate | None = None


@dataclass(frozen=True)
class DecideSiteInput:
    """What :func:`~noctis.research.briefings.decide_briefing` is called with, for one candidate."""

    toolbox: ResearchFacts
    ledger: SessionLedger
    strategy: str
    context_window: int
    mandate: Mandate | None = None


@dataclass(frozen=True)
class DiscoverSiteInput:
    """What :func:`~noctis.research.briefings.discover_briefing` is called with.

    ``profile`` is the band profile the structural screen found no lake name for, and ``window``
    the history a fetch would cover — the spend context the ask is made inside.
    """

    toolbox: ResearchFacts
    ledger: SessionLedger
    thesis: str
    symbol_character: str
    profile: Mapping[str, str]
    context_window: int
    window: Mapping[str, Any] | None = None
    mandate: Mandate | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Renderers — thin forwarding adapters onto the production builders
# ─────────────────────────────────────────────────────────────────────────────
def render_formulate(site_input: FormulateSiteInput, spec: HarnessSpec) -> str:
    """Forward to :func:`~noctis.research.briefings.formulate_briefing`, unchanged.

    ``spec`` is part of the declared render signature and is deliberately unread: the composition
    ablations it carries are wired into the builders by the prompt-ablation epic. Until then a
    non-production spec renders exactly what production renders — an honest no-op, not a silent
    partial ablation that would make a bench record's claim untrue.
    """
    return formulate_briefing(
        site_input.toolbox,
        site_input.ledger,
        mandate=site_input.mandate,
        context_window=site_input.context_window,
    )


def render_decide(site_input: DecideSiteInput, spec: HarnessSpec) -> str:
    """Forward to :func:`~noctis.research.briefings.decide_briefing`, unchanged.

    ``spec`` is accepted and unread for now — see :func:`render_formulate`.
    """
    return decide_briefing(
        site_input.toolbox,
        site_input.ledger,
        site_input.strategy,
        mandate=site_input.mandate,
        context_window=site_input.context_window,
    )


def render_discover(site_input: DiscoverSiteInput, spec: HarnessSpec) -> str:
    """Forward to :func:`~noctis.research.briefings.discover_briefing`, unchanged.

    ``spec`` is accepted and unread for now — see :func:`render_formulate`.
    """
    return discover_briefing(
        site_input.toolbox,
        site_input.ledger,
        thesis=site_input.thesis,
        symbol_character=site_input.symbol_character,
        profile=site_input.profile,
        window=site_input.window,
        mandate=site_input.mandate,
        context_window=site_input.context_window,
    )


# ─────────────────────────────────────────────────────────────────────────────
# The declarations
# ─────────────────────────────────────────────────────────────────────────────
FORMULATE_SITE: AgentSite[FormulateSiteInput, FormulateOutput] = AgentSite(
    id="formulate",
    version=_INITIAL_VERSION,
    contract=FORMULATE_CONTRACT,
    render=render_formulate,
    knobs=FormulateKnobs,
    scorers=(),
)

# Decide declares the PRIMARY contract — the three-way ask (approve / reject / revise) every
# verdict episode starts from. The final-ask variant (#99), whose emit vocabulary drops ``revise``
# once the revise budget is spent, is the same site asking the same question with a narrower
# answer set, so it rides this declaration's ``version``: a change to *either* contract's shape
# bumps it. Both are named together in :data:`DECIDE_CONTRACTS`.
#
# The one filled ``scorers`` slot today: approval-side agreement over a re-run's verdict, scored
# against what the promotion gates recorded doing with that candidate (#209).
DECIDE_SITE: AgentSite[DecideSiteInput, DecideOutput] = AgentSite(
    id="decide",
    version=_INITIAL_VERSION,
    contract=DECIDE_CONTRACT,
    render=render_decide,
    knobs=DecideKnobs,
    scorers=(DECIDE_SCORER,),
    # The three axes a mined DECIDE corpus labels its cases on, which every reading of this site
    # splits its agreement figure by. Declared here because a site's facts live in its declaration,
    # and carried to the scoring pass by the runner (#306).
    difficulty_axes=DIFFICULTY_AXES,
)

# Every contract the decide site's version covers, primary first — the driver's own objects, so a
# harness can enumerate the variant without the declaration growing a second contract field for a
# case that exists at exactly one site.
DECIDE_CONTRACTS: tuple[EmitContract[DecideOutput], ...] = (DECIDE_CONTRACT, DECIDE_FINAL_CONTRACT)

DISCOVER_SITE: AgentSite[DiscoverSiteInput, DiscoverOutput] = AgentSite(
    id="discover",
    version=_INITIAL_VERSION,
    contract=DISCOVER_CONTRACT,
    render=render_discover,
    knobs=DiscoverKnobs,
    scorers=(),
)

# The episodic sites in protocol order — formulate proposes, discover finds it symbols, decide
# rules on the evidence. Listed here so the registry adds them as one line.
EPISODIC_SITES: tuple[AgentSite[Any, Any], ...] = (FORMULATE_SITE, DECIDE_SITE, DISCOVER_SITE)
