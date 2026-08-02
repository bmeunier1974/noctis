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
    "CORPUS_DIRNAME",
    "DEFAULT_CONFIG_ID",
    "SITE_ASKS",
    "SITE_CORPORA",
    "SPLIT_WORDS",
    "BenchSeams",
    "LiveModelUnavailable",
    "SiteAsk",
    "SiteCorpus",
    "bench_root",
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
    "select_split",
    "site_ask",
    "site_corpus",
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
class SiteAsk:
    """How one site is asked: the input adapter, and the framing a live call needs.

    ``site_input`` is a factory over the resolved settings (the bench-wide facts an adapter closes
    over — DECIDE's context window, say) returning the per-case adapter the runner's ``site_input``
    seam takes. ``system`` and ``contract`` are the live half: production's own role framing, and
    the emit contract this case's reply is validated through (a callable, because DECIDE's frozen
    cases choose between the primary and the revise-less final ask). Both ``None`` means the site
    can be planned and rendered but not yet asked live — refused by name, never improvised.
    """

    site_input: Callable[[Any], Callable[[Case], Any]] = payload_input
    system: str | None = None
    contract: Callable[[AgentSite[Any, Any], Case], EmitContract[Any] | None] | None = None

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


#: One entry per site that needs more than its payload — the declaration table this module's
#: docstring describes. A site absent from it is asked with :func:`payload_input` and cannot be
#: asked live until its row lands here.
SITE_ASKS: Mapping[str, SiteAsk] = {
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
    """
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
