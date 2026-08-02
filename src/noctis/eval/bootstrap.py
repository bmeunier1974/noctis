"""The eval layer's own composition root — where a bench's collaborators are assembled, once.

:mod:`noctis.bootstrap` is the engine's composition root and it may never import this layer (the
one-way boundary :mod:`noctis.eval.guard` enforces). A bench still needs the same discipline —
*assemble sessions in one place, not by hand in a command body* — so this module is that place for
the other side of the line: the ``bench`` verbs state flags, this module turns them into a
:class:`~noctis.eval.runner.BenchRunner`, and the engine's builders are imported freely because the
arrow points this way.

**The site-input lookup, and why it is a mapping rather than a branch.** The runner renders a case
through the site's *declared* renderer, but a renderer takes the site's own input type — DECIDE's
takes a toolbox, a ledger and a strategy name (:mod:`noctis.eval.decide_site` reconstructs all
three from a frozen case), while a site whose renderer reads a plain mapping needs nothing at all.
The runner refuses to guess (``site_input`` is injected, never inferred), so somebody has to know
which adapter belongs to which site: :data:`SITE_ASKS` is that somebody. It is a **declaration
table**, exactly like the site registry beside it — one entry per site that needs more than its
payload, looked up by id — so the verb that drives a bench contains no site's name at all and a
site with no entry runs through the same code path on :func:`payload_input`. Adding a site's
adapter is editing this table, in a reviewable diff.

**The site-corpus lookup, its twin one artifact earlier.** A corpus is *loaded* before it is asked,
and not every site's cases sit in the same shape: most are a flat directory the generic provider
reads, while the coder's are partitioned into bucket directories and labelled against a closed set
of axis levels. :data:`SITE_CORPORA` is the declaration table for that — one row per site whose
corpus is not the flat default, naming the reader that serves it and the vocabulary an *absence* is
reported against — so :func:`load_corpus` has no branch on a site id in it and ``bench corpus``
reports a coder corpus and a mined DECIDE one down one code path.

**The site-tier lookup, its third sibling: a named population, declared as data.** A whole corpus is
the measurement; a **tier** is the faster question asked before spending on one. :data:`SITE_TIERS`
is one row per site per named tier, and a tier *is* its case ids — a reviewable list rather than a
predicate — so what ``--tier smoke`` selects can be read off the table without running anything and
a tier that has quietly stopped being twelve cases fails a test rather than publishing its name over
a different population. Two rules keep it honest: a tier is applied to an **already-dealt** corpus,
so no case's tuning/holdout half moves by being asked about in a smaller group
(:meth:`CaseTier.select`); and a tier and a ``--split`` are two ways of naming a population, so
stating both is **refused** rather than intersected (:func:`select_population`) — filtering a
declared twelve down to its holdout would publish the word ``smoke`` over something else entirely.

**The live model call, and what it honestly is today.** A live bench asks the configured model
through the engine's own provider-neutral seam: :func:`~noctis.research.llm.client_for` builds the
client, :class:`~noctis.research.episode.EpisodeRunner` makes the forced structured-emit call, and
the site's declared emit contract validates the reply — the same three objects a research session
uses, so a benched ask fails and retries the way a real one does. Two facts a reader should not
have to discover:

* the **system framing** is production's own text, quoted per site in :data:`SITE_ASKS`, because a
  site declaration carries its briefing renderer and not the one line of role framing above it. A
  site whose ask is not declared here is **refused** for a live run rather than asked under an
  invented system prompt — a benchmark measuring a prompt nobody ships is worse than no number;
* a **dry run never reaches any of it.** :func:`live_attempt` is called only on the live path, so
  ``bench run --dry-run`` needs no key, no ``[llm]`` extra and no network, and the client that is
  never built cannot spend anything.

Every collaborator is injectable through :class:`BenchSeams` — the attempt callable, the site
registry, the ask table, the resolved identity — which is what lets the whole suite drive both
verbs end to end over a stub site with no network at all.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from noctis.eval.case import Case, CaseProvider, Split
from noctis.eval.case_provider import YamlCaseProvider
from noctis.eval.corpus import Corpus
from noctis.eval.corpus_report import CorpusVocabulary
from noctis.eval.harness import HarnessSpec
from noctis.eval.identity import SiteIdentity
from noctis.eval.metrics import AttemptOutcome
from noctis.eval.pool import PooledExecutor
from noctis.eval.record import ModelConfig
from noctis.eval.registry import site
from noctis.eval.runner import (
    Attempt,
    AttemptFn,
    AttemptRequest,
    BenchRunner,
    JobExecutor,
    SequentialExecutor,
    bench_root,
)
from noctis.eval.site import AgentSite
from noctis.research.episode import EmitContract
from noctis.research.pricing import USAGE_FIELDS

__all__ = [
    "ALL_SPLITS",
    "CODER_SMOKE_CASES",
    "CORPUS_DIRNAME",
    "DEFAULT_CONFIG_ID",
    "SEQUENTIAL_WORKERS",
    "SITE_ASKS",
    "SITE_CORPORA",
    "SITE_TIERS",
    "SMOKE_TIER",
    "SPLIT_WORDS",
    "TIER_WORKER_CAP",
    "BenchSeams",
    "CaseTier",
    "LiveAsk",
    "LiveModelUnavailable",
    "Population",
    "SiteAsk",
    "SiteCorpus",
    "bench_root",
    "bench_width",
    "build_bench_runner",
    "build_executor",
    "cases_root",
    "configs_for",
    "corpus_provider",
    "flat_provider",
    "live_attempt",
    "load_cases",
    "load_corpus",
    "payload_input",
    "select_population",
    "select_split",
    "select_tier",
    "site_ask",
    "site_corpus",
    "site_tiers",
    "site_vocabulary",
]

#: The workspace-level corpus root, beside ``bench/`` and the shared lake: cases are run-neutral
#: input, and the DECIDE miner writes into exactly this tree.
CORPUS_DIRNAME = "cases"

#: The word that names *both* halves of a corpus. Spelled out rather than left as an absent flag,
#: because "the whole corpus" and "nobody said" read identically on a command line and must not.
ALL_SPLITS = "all"

#: Every word ``--split`` accepts, in the order a refusal lists them.
SPLIT_WORDS: tuple[str, ...] = (Split.TUNING.value, Split.HOLDOUT.value, ALL_SPLITS)

#: The one configuration id a single-model bench run uses. The *model* rides in the configuration's
#: ``requested_model``, never in its id: an id names a directory, and a ``provider/model`` alias
#: carries a separator the runner refuses (rightly) as an artifact name.
DEFAULT_CONFIG_ID = "default"


class LiveModelUnavailable(RuntimeError):
    """No live ask could be made: no client can be built, or the site declares no live ask."""


# ── the corpus ────────────────────────────────────────────────────────────────────────────


def cases_root(workspace_dir: Path | str) -> Path:
    """The corpus root of one workspace: ``<workspace>/cases``, one directory per site.

    Workspace-level for the same reason the bench area is: a corpus is a population, not one run's
    trajectory, and two runs in a workspace measure the same cases the way they read one lake.
    """
    return Path(workspace_dir) / CORPUS_DIRNAME


def flat_provider(
    cases_root: Path, registry: Mapping[str, AgentSite[Any, Any]] | None
) -> CaseProvider:
    """The default reader: one directory per site, one ``*.yaml`` per case, validated file by file.

    A site whose corpus is a flat directory needs nothing else, which is what makes the corpus verbs
    site-agnostic — the sites whose layout differs declare a reader in :data:`SITE_CORPORA`.
    """
    return YamlCaseProvider(cases_root=cases_root, registry=registry)


@dataclass(frozen=True)
class SiteCorpus:
    """How one site's corpus is read, and what vocabulary its labels are declared against.

    The corpus twin of :class:`SiteAsk`, and a **declaration table** for the same reason: most
    corpora are a flat directory of YAML files that the generic provider reads, but the coder's is
    bucket-partitioned (:class:`~noctis.eval.coder_corpus.CoderCaseProvider`) and its axes are a
    closed vocabulary (:data:`~noctis.eval.coder_case.AXIS_LEVELS`). ``provider`` is a factory over
    the cases root and the site registry; ``vocabulary`` is a factory (deferred, because the coder
    row pulls in the author engine) returning what an absence is named against — a site with no row
    gets the flat reader and no vocabulary at all, and reports over exactly what its cases carry.
    """

    provider: Callable[[Path, Mapping[str, AgentSite[Any, Any]] | None], CaseProvider] = (
        flat_provider
    )
    vocabulary: Callable[[], CorpusVocabulary] = CorpusVocabulary


def _coder_provider(
    cases_root: Path, registry: Mapping[str, AgentSite[Any, Any]] | None
) -> CaseProvider:
    """The coder corpus's own reader: one directory per bucket, validated by the coder schema."""
    from noctis.eval.coder_corpus import CoderCaseProvider

    return CoderCaseProvider(cases_root=cases_root, registry=registry)


def _coder_vocabulary() -> CorpusVocabulary:
    """What the coder site declares its cases are labelled with: four buckets, seven axes.

    The buckets are ordered what-ships-first (``edge``, ``canary``, then the two an operator mines
    for themselves), because that is the order a reader of a *committed* corpus wants them in.
    """
    from noctis.eval.coder_case import AXIS_LEVELS, Axis
    from noctis.eval.coder_corpus import COMMITTED_BUCKETS, LOCAL_BUCKETS, bucket_of

    return CorpusVocabulary(
        bucket_of=lambda case: bucket_of(case).value,
        buckets=tuple(bucket.value for bucket in (*COMMITTED_BUCKETS, *LOCAL_BUCKETS)),
        axis_levels={axis.value: AXIS_LEVELS[axis] for axis in Axis},
    )


#: One entry per site whose corpus is not a flat directory of cases labelled with free-form axes.
#: A site absent from it is read by :class:`~noctis.eval.case_provider.YamlCaseProvider` and
#: reported over the labels its own cases carry — adding a row is a reviewable diff, like
#: :data:`SITE_ASKS` beside it.
SITE_CORPORA: Mapping[str, SiteCorpus] = {
    "coder": SiteCorpus(provider=_coder_provider, vocabulary=_coder_vocabulary),
}


def site_corpus(site_id: str, corpora: Mapping[str, SiteCorpus] | None = None) -> SiteCorpus:
    """How ``site_id``'s corpus is read — its declared row, or the generic flat default.

    The table is passed in rather than reached for globally, exactly as :func:`site_ask` takes its
    own, so a harness can index a scratch set without a global to reset.
    """
    table = SITE_CORPORA if corpora is None else corpora
    return table.get(site_id, SiteCorpus())


def site_vocabulary(
    site_id: str, corpora: Mapping[str, SiteCorpus] | None = None
) -> CorpusVocabulary:
    """The buckets and axis levels ``site_id`` declares — empty for a site that declares none."""
    return site_corpus(site_id, corpora).vocabulary()


def corpus_provider(
    site_id: str,
    *,
    cases_root: Path | str,
    registry: Mapping[str, AgentSite[Any, Any]] | None = None,
    corpora: Mapping[str, SiteCorpus] | None = None,
) -> CaseProvider:
    """The reader that serves one site's corpus — the site's declared loader, or the generic one."""
    return site_corpus(site_id, corpora).provider(Path(cases_root), registry)


def load_cases(
    site_id: str,
    *,
    cases_root: Path | str,
    registry: Mapping[str, AgentSite[Any, Any]] | None = None,
    corpora: Mapping[str, SiteCorpus] | None = None,
) -> tuple[Case, ...]:
    """Every case a site declares, validated file by file, exactly as its files assign them.

    Undealt on purpose: this is what a reader wanting to know which cases carry their own
    ``split:`` needs, and :func:`load_corpus` is the one line that turns it into a dealt corpus.
    """
    return corpus_provider(site_id, cases_root=cases_root, registry=registry, corpora=corpora).load(
        site_id
    )


def load_corpus(
    site_id: str,
    *,
    cases_root: Path | str,
    registry: Mapping[str, AgentSite[Any, Any]] | None = None,
    corpora: Mapping[str, SiteCorpus] | None = None,
) -> Corpus:
    """One site's corpus, loaded through its provider and split once — for *any* site.

    Which provider that is comes from :data:`SITE_CORPORA` rather than from a branch here, so a
    mined DECIDE corpus, a hand-written flat one and the coder's bucket-partitioned one are all
    admitted on the same terms: the site's own validation, then
    :class:`~noctis.eval.corpus.Corpus` deals the split.
    """
    cases = load_cases(site_id, cases_root=cases_root, registry=registry, corpora=corpora)
    return Corpus(site_id=site_id, cases=cases)


def select_split(word: str | None) -> Split | None:
    """The half of a corpus a word names — ``None`` for the whole of it, a refusal for anything.

    ``None`` is the runner's own "both halves"; :data:`ALL_SPLITS` is how an operator says it out
    loud. A word neither is refused naming all three, because silently measuring the whole corpus
    when somebody asked for the holdout is the one mistake this flag must not make.
    """
    if word is None or word == ALL_SPLITS:
        return None
    for split in Split:
        if word == split.value:
            return split
    raise ValueError(
        f"--split {word!r} names no half of a corpus — it takes {', '.join(SPLIT_WORDS)}"
    )


# ── the declared tiers ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CaseTier:
    """A named subset of one site's corpus, declared by case id — a population, spelled out.

    A tier is **data**: the ids are the selection rule, so what ``--tier smoke`` measures is read
    off :data:`SITE_TIERS` rather than inferred from a predicate nobody can evaluate by eye, and
    ``rationale`` says in prose why *these* cases (the table is the review). Two consequences are
    deliberate:

    * **the size is a promise.** A tier that names a case the corpus no longer holds is refused
      naming it, because publishing a tier's name over eleven of its twelve cases would report one
      population under another's label;
    * **the split is not re-decided.** :meth:`select` takes cases out of an *already-dealt* corpus,
      every one of which carries its frozen half, so the corpus it returns re-deals nothing (see
      :mod:`noctis.eval.corpus` — a frozen case passes through the deal untouched). A tuning case
      cannot become a holdout one by being asked about in a smaller group.
    """

    name: str
    case_ids: tuple[str, ...]
    rationale: str

    def __len__(self) -> int:
        """How many cases this tier declares."""
        return len(self.case_ids)

    def select(self, corpus: Corpus) -> Corpus:
        """This tier's cases, out of a dealt corpus — or a refusal naming the ones that are gone."""
        held = {case.case_id: case for case in corpus.cases}
        missing = [case_id for case_id in self.case_ids if case_id not in held]
        if missing:
            raise ValueError(
                f"--tier {self.name!r} declares {len(self)} case(s), and this corpus holds none "
                f"named {', '.join(repr(case_id) for case_id in missing)} — a tier is a declared "
                "population, so a partial one would publish its name over another measurement"
            )
        return Corpus(
            site_id=corpus.site_id, cases=tuple(held[case_id] for case_id in self.case_ids)
        )


#: The coder smoke tier's twelve cases, and the rule that chose them: **every canary the corpus
#: ships** (six briefs so plain that a red one indicts the harness rather than the model) **plus the
#: smallest set of edge cases that, together with those canaries, exercises every level of all seven
#: difficulty axes**. The canaries alone reach only the easy end (``scratch``/``authored``/
#: ``rolling``/``trivial``/``narrow``/``indicators``), so each edge case below is here for the
#: levels it is the cheapest cover for:
#:
#: * ``edge-reference-adapt-donchian-to-shorts`` — ``composition_mode: reference``, ``latched``;
#: * ``edge-revision-shrink-warmup-for-fixed-oracle`` — ``composition_mode: revision``,
#:   ``oracle_mode: fixed_spec``;
#: * ``edge-higher-timeframe-daily-regime-filter`` — ``warmup_arithmetic: higher_timeframe``;
#: * ``edge-stateless-bar-shape-close-strength`` — ``state_complexity: stateless``,
#:   ``api_surface: bars_only``;
#: * ``edge-scale-free-percentile-return-rank`` — ``no_trade_tape: scale_free`` (the feasibility
#:   rules' hardest case), ``param_space_breadth: moderate``;
#: * ``edge-broad-param-space-multi-filter-breakout`` — ``param_space_breadth: broad``,
#:   ``api_surface: exits``.
#:
#: Twelve exactly, and the coverage claim is asserted over the committed corpus by enumeration
#: (``tests/test_eval_bench_tier.py``), so a case whose labels move breaks the build rather than
#: quietly thinning the tier.
CODER_SMOKE_CASES: tuple[str, ...] = (
    "canary-close-above-single-average",
    "canary-ema-crossover",
    "canary-high-price-crossover",
    "canary-inverted-crossover",
    "canary-long-short-crossover",
    "canary-slower-slow-average",
    "edge-broad-param-space-multi-filter-breakout",
    "edge-higher-timeframe-daily-regime-filter",
    "edge-reference-adapt-donchian-to-shorts",
    "edge-revision-shrink-warmup-for-fixed-oracle",
    "edge-scale-free-percentile-return-rank",
    "edge-stateless-bar-shape-close-strength",
)

#: The coder's fast tier, as an operator names it on the command line.
SMOKE_TIER = CaseTier(
    name="smoke",
    case_ids=CODER_SMOKE_CASES,
    rationale=(
        "every canary plus the six edge cases that complete the axis coverage — twelve jobs, the "
        "smallest population that still touches every level of all seven difficulty axes"
    ),
)

#: One row per site that declares tiers, keyed by the word ``--tier`` takes. A site absent from it
#: has no tiers at all and says so rather than accepting a name it cannot honour.
SITE_TIERS: Mapping[str, Mapping[str, CaseTier]] = {"coder": {SMOKE_TIER.name: SMOKE_TIER}}


def site_tiers(
    site_id: str, tiers: Mapping[str, Mapping[str, CaseTier]] | None = None
) -> Mapping[str, CaseTier]:
    """Every tier ``site_id`` declares — empty for a site that declares none."""
    table = SITE_TIERS if tiers is None else tiers
    return table.get(site_id, {})


def select_tier(
    site_id: str, name: str | None, tiers: Mapping[str, Mapping[str, CaseTier]] | None = None
) -> CaseTier | None:
    """The tier a word names for one site — ``None`` for no tier, a refusal for anything else.

    The refusal names the tiers that site declares (or says it declares none), because "unknown
    tier" is only useful beside the list an operator could have typed instead.
    """
    if name is None:
        return None
    declared = site_tiers(site_id, tiers)
    if name in declared:
        return declared[name]
    if not declared:
        raise ValueError(
            f"--tier {name!r}: site {site_id!r} declares no tiers — drop --tier to measure its "
            "whole corpus"
        )
    raise ValueError(
        f"--tier {name!r} names no tier of site {site_id!r} — it declares "
        f"{', '.join(sorted(declared))}"
    )


@dataclass(frozen=True)
class Population:
    """What one bench measures: a half of a corpus, or a declared tier of it — never both.

    Both fields empty is the whole corpus, which is what a bench measured before tiers existed.
    """

    split: Split | None = None
    tier: CaseTier | None = None

    def of(self, corpus: Corpus) -> Corpus:
        """The corpus this bench runs over — the tier's cases, or the loaded corpus unchanged.

        The ``--split`` half is *not* applied here: the runner takes it as its own argument, so a
        record still states which half it measured rather than inferring it from a case count.
        """
        return corpus if self.tier is None else self.tier.select(corpus)


def select_population(
    site_id: str,
    *,
    split: str | None,
    tier: str | None,
    tiers: Mapping[str, Mapping[str, CaseTier]] | None = None,
) -> Population:
    """Resolve the two flags that name a population into one value, refusing every ambiguity.

    A ``--split`` word names a *half* of a corpus and a ``--tier`` names a *declared subset* of it:
    two answers to one question, so stating both is refused naming both flags rather than silently
    intersected. Intersecting them is the tempting move and the dishonest one — a twelve-case tier
    filtered to its holdout is a handful of cases wearing the word ``smoke``, and the number that
    came out would be compared against a tier nobody ran. The whole-corpus word
    (:data:`ALL_SPLITS`, and the absence a bare invocation leaves) filters nothing, so it composes
    with a tier freely.
    """
    named = select_tier(site_id, tier, tiers)
    selected = select_split(split)
    if named is not None and selected is not None:
        raise ValueError(
            f"--tier {named.name!r} and --split {selected.value!r} both name what to measure — "
            f"a tier is a declared population of {len(named)} case(s), and filtering it to one "
            "half would publish the tier's name over a different measurement. State one of them."
        )
    return Population(split=selected, tier=named)


# ── how each site is asked ────────────────────────────────────────────────────────────────


def payload_input(settings: Any) -> Callable[[Case], Any]:
    """The default adapter: a case's payload, exactly as its file declared it.

    A site whose renderer reads a mapping needs nothing else, which is what makes the verb
    site-agnostic — the sites that *do* need an adapter declare one in :data:`SITE_ASKS`.
    """

    def adapt(case: Case) -> Any:
        return case.payload

    return adapt


@dataclass(frozen=True)
class LiveAsk:
    """What a site's own live attempt maker is handed — the one place a bench's world is stated.

    A frozen record rather than three arguments, for the reason :class:`SiteAsk` is one: the day an
    attempt maker needs one more fact is a new field here, not a broken signature at the two places
    that build one. ``client`` is the seam a test injects; ``None`` means the maker builds its own.
    """

    settings: Any
    model: str | None = None
    client: Any = None


@dataclass(frozen=True)
class SiteAsk:
    """How one site is asked: the input adapter, and what a live call needs.

    ``site_input`` is a factory over the resolved settings (the bench-wide facts an adapter closes
    over — DECIDE's context window, say) returning the per-case adapter the runner's ``site_input``
    seam takes. ``system`` and ``contract`` are the live half of a site whose answer a **forced
    structured emit** carries: production's own role framing, and the emit contract this case's
    reply is validated through (a callable, because DECIDE's frozen cases choose between the primary
    and the revise-less final ask). All ``None`` means the site can be planned and rendered but not
    yet asked live — refused by name, never improvised.

    ``attempt`` is for the sites a forced emit is simply *not* how they are asked. The coder's ask
    is an authoring job whose answer the fresh-subprocess write gate judges — no schema, no
    contract, and a throwaway library to build first — so its row declares its own maker
    (:func:`noctis.eval.coder_site.coder_attempt`) and :func:`live_attempt` hands the whole live
    call over rather than approximating it. A row that declares one is asked through it and nothing
    else; a row that does not takes the generic emit path exactly as before.
    """

    site_input: Callable[[Any], Callable[[Case], Any]] = payload_input
    system: str | None = None
    contract: Callable[[AgentSite[Any, Any], Case], EmitContract[Any] | None] | None = None
    attempt: Callable[[LiveAsk], AttemptFn] | None = None

    def emit_contract(
        self, declaration: AgentSite[Any, Any], case: Case
    ) -> EmitContract[Any] | None:
        """The contract this case's reply is read through — the site's own unless one is chosen."""
        if self.contract is None:
            return declaration.contract
        return self.contract(declaration, case)


def _decide_input(settings: Any) -> Callable[[Case], Any]:
    """DECIDE's adapter: a frozen case back into the production briefing's three inputs.

    The context window is the session's own — a briefing is trimmed to fit it, so a bench that
    measured under another window measured another prompt.
    """
    from noctis.eval.decide_site import decide_site_input

    return decide_site_input(context_window=_context_window(settings))


def _decide_contract(declaration: AgentSite[Any, Any], case: Case) -> EmitContract[Any] | None:
    """DECIDE's contract for one case: the final (revise-less) ask when history spent its cap."""
    from noctis.eval.decide_site import contract_for

    return contract_for(case)


def _decide_system() -> str:
    """The DECIDE role framing, quoted from the driver that ships it rather than restated."""
    from noctis.research.driver import _DECIDE_SYSTEM

    return _DECIDE_SYSTEM


def _coder_input(settings: Any) -> Callable[[Case], Any]:
    """The coder's adapter: one case as the authoring job the site's renderer takes.

    The engine it carries reads the *committed* seed library (the shipped ``TEMPLATE.py`` and worked
    example are half the coder's prompt), so a rendered preview is what a real install sends.
    """
    from noctis.eval.coder_site import coder_site_input

    return coder_site_input(settings)


def _coder_attempt(ask: LiveAsk) -> AttemptFn:
    """The coder's live ask: one whole authoring job per attempt, judged by the write gate.

    Assembled here, in this layer's composition root, exactly as the generic live call is: the
    clients are built once per bench (with the coder's own dials — thinking, sampling — because a
    bench asking the same model without them measures a configuration nobody ships), and the job
    itself lives in :mod:`noctis.eval.coder_site`.
    """
    from noctis.eval.coder_site import coder_attempt, coder_clients

    clients = coder_clients(ask.settings, model=ask.model, client=ask.client)
    return coder_attempt(
        ask.settings,
        client=clients.client,
        model=clients.model,
        fallback_client=clients.fallback,
        fallback_model=clients.fallback_model,
    )


def _coder_system() -> str:
    """The coder's role framing, quoted from the engine that ships it rather than restated.

    Declared for the same reason DECIDE's is — a site declaration carries its briefing renderer and
    not the line of role framing above it — with one honest caveat: the coder's *live* ask composes
    its whole system prompt inside the engine (role framing, contract sheet, feasibility rules,
    template, worked example), so this is what the framing IS, never a second copy anything sends.
    """
    from noctis.research.author import _ROLE_RULES

    return _ROLE_RULES


#: One entry per site that needs more than its payload — the declaration table this module's
#: docstring describes. A site absent from it is asked with :func:`payload_input` and cannot be
#: asked live until its row lands here.
SITE_ASKS: Mapping[str, SiteAsk] = {
    "coder": SiteAsk(
        site_input=_coder_input,
        system=_coder_system(),
        # No emit contract, on purpose: the write gate is this site's judge, so the coder declares
        # its own live maker instead of a schema (see :class:`SiteAsk`).
        contract=None,
        attempt=_coder_attempt,
    ),
    "decide": SiteAsk(site_input=_decide_input, system=_decide_system(), contract=_decide_contract),
}


def site_ask(site_id: str, asks: Mapping[str, SiteAsk] | None = None) -> SiteAsk:
    """How ``site_id`` is asked — its declared row, or the payload default.

    The table is passed in rather than reached for globally, exactly as every other eval-layer
    lookup takes its registry, so a harness can index a scratch set without a global to reset.
    """
    table = SITE_ASKS if asks is None else asks
    return table.get(site_id, SiteAsk())


def _context_window(settings: Any) -> int:
    """The window the briefings are budgeted against — the session's, or the episodic default.

    Read from the engine's composition root by import rather than restated: a bench that trimmed
    to a different budget than a session does would be measuring a different prompt. The import is
    deferred because it pulls the whole engine root in for one integer.
    """
    from noctis.bootstrap import _EPISODIC_CONTEXT_WINDOW

    return int(getattr(settings.research.agent, "context_window", None) or _EPISODIC_CONTEXT_WINDOW)


# ── the live model call ───────────────────────────────────────────────────────────────────


def live_attempt(
    settings: Any,
    *,
    site_id: str | None = None,
    model: str | None = None,
    client: Any = None,
    registry: Mapping[str, AgentSite[Any, Any]] | None = None,
    asks: Mapping[str, SiteAsk] | None = None,
) -> AttemptFn:
    """The attempt callable that really asks the configured model, through the engine's own seam.

    Built **once** per bench (one client, reused across jobs) and refused up front when no client
    can be built, so an operator learns that the ``[llm]`` extra is missing before a corpus is
    walked rather than once per case. ``client`` is the seam a test injects; production leaves it
    ``None`` and gets :func:`~noctis.research.llm.client_for`'s.

    ``site_id`` is what a bench already knows — it measures exactly one site — and it is what lets a
    site whose ask is not a forced structured emit declare its own maker (:attr:`SiteAsk.attempt`).
    Left unstated, every case takes the generic emit path, resolved per case as before.
    """
    if site_id is not None:
        declared = site_ask(site_id, asks).attempt
        if declared is not None:
            return declared(LiveAsk(settings=settings, model=model, client=client))
    from noctis.research.llm import (
        _client_blocked,
        client_for,
        provider_of,
        resolved_research_model,
    )

    alias = model or resolved_research_model(settings)
    if client is None:
        blocked = _client_blocked(provider_of(alias), settings)
        if blocked is not None:
            raise LiveModelUnavailable(f"no model client for {alias!r} — {blocked}")
        client = client_for(settings, alias)
    if client is None:  # pragma: no cover - the gate above is the one that fires
        raise LiveModelUnavailable(f"no model client for {alias!r}")

    def attempt(request: AttemptRequest) -> Attempt:
        return _ask_live(client, settings, request, alias=alias, registry=registry, asks=asks)

    return attempt


def _ask_live(
    client: Any,
    settings: Any,
    request: AttemptRequest,
    *,
    alias: str,
    registry: Mapping[str, AgentSite[Any, Any]] | None,
    asks: Mapping[str, SiteAsk] | None,
) -> Attempt:
    """One forced structured-emit call — production's episode runner, over the bench's prompt."""
    from noctis.research.episode import EpisodeRunner

    ask = site_ask(request.case.site_id, asks)
    declaration = site(request.case.site_id, registry)
    contract = ask.emit_contract(declaration, request.case)
    if ask.system is None or contract is None:
        raise LiveModelUnavailable(
            f"site {declaration.id!r} declares no live ask — its system framing and emit contract "
            "belong in SITE_ASKS, and a benchmark never invents a prompt production does not ship"
        )
    # The session's own dials: the misfire-retry budget and the output ceiling. A bench asking under
    # a different retry policy would report a pass rate production never has.
    agent = settings.research.agent
    retries = int(getattr(agent, "episode_retries", 0) or 0)
    ceiling = int(getattr(agent, "max_tokens", None) or 0)
    runner = (
        EpisodeRunner(client=client, retries=retries, max_tokens=ceiling)
        if ceiling
        else EpisodeRunner(client=client, retries=retries)
    )
    started = time.perf_counter()
    episode = runner.run(contract=contract, system=ask.system, briefing=request.prompt)
    elapsed = round(time.perf_counter() - started, 6)
    # Only the four billed fields, and only the ones the backend really reported: an absent field
    # stays absent (unpriceable), because a zero would price an unknown as free.
    spent = {
        name: int(count)
        for name, count in (episode.usage or {}).items()
        if name in USAGE_FIELDS and count is not None
    }
    return Attempt(
        outcome=AttemptOutcome(
            passed=episode.ok,
            model=alias,
            seconds=elapsed,
            error=None if episode.ok else (episode.note or episode.outcome),
            **spent,
        ),
        output=_emitted(episode.value),
        served_model=episode.served_model or None,
    )


def _emitted(value: Any) -> str | None:
    """The emitted record as the artifact a bench retains — canonical JSON where it is one."""
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return json.dumps(asdict(value), sort_keys=True, default=str)
    return str(value)


# ── the assembly ──────────────────────────────────────────────────────────────────────────


def build_executor(workers: int) -> JobExecutor:
    """How the jobs are worked: one at a time, or on a pool of that many workers.

    One worker is the sequential executor rather than a pool of one — a pool that forks a single
    worker buys nothing and adds every failure mode a pool has.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    return SequentialExecutor() if workers == 1 else PooledExecutor(workers=workers)


#: What an untiered bench does when nobody states a width: one job at a time. A bench spends real
#: money against somebody's rate limit, so widening it stays an explicit act.
SEQUENTIAL_WORKERS = 1

#: How wide a **tiered** run opens the pool when nobody states a width — the derivation, stated so
#: it can be argued with rather than tuned. A tier exists to answer a question in minutes: the coder
#: smoke tier is twelve independent jobs (one case, one rep, one configuration each), and a job is a
#: provider round trip plus a fresh-subprocess write gate, so the wall clock is
#: ``ceil(jobs / width) × slowest job``. At six, twelve jobs are **two waves** — a five-minute
#: target therefore asks that one authoring job finish inside about two and a half minutes, which
#: is the budget a job already has (per-attempt timeout × retry budget). The cap is not
#: the job count on purpose: twelve concurrent completions is a rate-limit decision an operator
#: should take deliberately with ``--workers``, and this is the width the shipped tier is sized at.
TIER_WORKER_CAP = 6


def bench_width(stated: int | None, population: Population, *, cases: int) -> int:
    """How wide this bench's jobs are worked: what an operator stated, or what the tier derives.

    A stated ``--workers`` always wins, including on a tiered run — the derivation is a default, not
    a policy. Untiered and unstated stays :data:`SEQUENTIAL_WORKERS`, which is what a bench did
    before tiers existed; a tier smaller than :data:`TIER_WORKER_CAP` opens only as many workers as
    it has cases, because a pool wider than its work is idle processes and one more failure mode.
    """
    if stated is not None:
        return stated
    if population.tier is None:
        return SEQUENTIAL_WORKERS
    return max(SEQUENTIAL_WORKERS, min(cases, TIER_WORKER_CAP))


def configs_for(model: str | None) -> tuple[ModelConfig, ...]:
    """The one configuration a single-model bench runs under — see :data:`DEFAULT_CONFIG_ID`."""
    return (ModelConfig(config_id=DEFAULT_CONFIG_ID, requested_model=model),)


@dataclass(frozen=True)
class BenchSeams:
    """Everything a bench's assembly takes from outside the operator's own flags.

    One injection point rather than four keyword arguments threaded through the verb: a test
    (or a future harness) hands over a stub attempt, a scratch registry, its own ask table and a
    resolved identity, and the command body passes none of them.
    """

    attempt: AttemptFn | None = None
    registry: Mapping[str, AgentSite[Any, Any]] | None = None
    asks: Mapping[str, SiteAsk] | None = None
    tiers: Mapping[str, Mapping[str, CaseTier]] | None = None
    identity: SiteIdentity | None = None


def build_bench_runner(
    settings: Any,
    *,
    site_id: str,
    attempt: AttemptFn,
    workers: int = 1,
    label: str | None = None,
    harness: HarnessSpec | None = None,
    seams: BenchSeams | None = None,
) -> BenchRunner:
    """One assembled runner: the bench area, the execution seam, and the site's input adapter.

    The single place a bench's collaborators meet, so both verbs — and anything that grows beside
    them — get the same bench area, the same adapter lookup and the same defaults.
    """
    chosen = seams or BenchSeams()
    return BenchRunner(
        bench_root=bench_root(settings.workspace_dir),
        attempt=attempt,
        registry=chosen.registry,
        harness=harness if harness is not None else HarnessSpec(),
        executor=build_executor(workers),
        identity=chosen.identity,
        label=label,
        site_input=site_ask(site_id, chosen.asks).site_input(settings),
    )
