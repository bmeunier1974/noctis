"""The coder site's runner integration: a throwaway library per job, and the write gate as scorer.

Every other benchmarkable site answers a question; the coder *writes a file*, and a file has to land
somewhere. That is the whole difficulty this module exists for, and it is answered in one sentence:
**a coder job builds its own three-tier library inside the throwaway working directory the runner
handed it** (:attr:`~noctis.eval.runner.AttemptRequest.workdir`), authors into it, and is judged by
the same fresh-subprocess write gate a research session is judged by. Nothing else on the machine
can see that library, and nothing that library does can be seen anywhere else.

**The tiers, and why only two of them are fresh.** :func:`job_library` points the ``seeds`` tier at
the **real committed** ``strategies/`` directory — in place, never copied — because the seeds are
read-only input by construction: :func:`~noctis.strategies.library.write_strategy` writes only into
the working tier, and a rewrite of a *seed's own name* is redirected there too, so a job that
authors ``sma_crossover`` leaves the committed file byte-identical and lands its own version in
``<workdir>/strategies/__tmp/``. Copying the seeds would measure a different install (the coder's
prompt folds the shipped ``TEMPLATE.py`` and worked example in), and would quietly hide the day that
redirection broke. The two writable tiers *are* fresh and empty per job, so one case can neither
read nor overwrite another's drafts, and the champion tier exists only so the gate's
champion-immutability check has something to look at.

**The score is the gate's verdict, and there is no second opinion.** A validated file lands (the
job passed) or an :class:`~noctis.research.author.AuthoringError` falls out (the job failed,
carrying the last gate error). It never inspects the source, never re-runs a check the gate ran,
and never softens one: the gate is the arbiter of quality by design, and a benchmark that graded on
anything else would be measuring its own opinion.

**Two levels of attempt, and the record says so.** The bench runner thinks in *attempts per rep*;
the authoring engine thinks in *private retries inside one job*. They are reconciled the honest way
rather than flattened: **one runner attempt is one whole authoring job**, retry budget included, and
each of the engine's internal attempts lands as an :class:`AttemptRecord` in the
:class:`JobRecord` this module returns as that attempt's output. So the runner's own aggregate
("did this job pass?") stays the runner's, and the finer reading a later story computes —
first-attempt pass rate, what retries rescued, what escalation bought — is derivable from the
retained record without the runner learning a word of this site's vocabulary. A bench that set
``max_attempts`` above one would be asking the *same case* twice with a fresh library each time,
which is a rep, not a retry — so the retry budget lives where production keeps it: on the engine.

**The degenerate-pass detectors run here, once, at the end of a job that passed.** They ask the
question the write gate cannot (*how* was this pass won — see :mod:`noctis.eval.coder_detectors`),
and answering it needs the loaded class and a replay, which is state only the job still holds: the
validated file lives in a throwaway library this job is about to abandon. So the run happens where
the file is (bounded, per job, only on a pass) and the findings are **stamped into the**
:class:`JobRecord`, which keeps the site scorer a pure function over retained records instead of a
reader that re-imports strategy files. Nothing about a finding touches the verdict: a job that drew
two warnings passed exactly as hard as one that drew none.

**What a job retains, inside its own workdir.** The exact prompt the engine sent
(:data:`PROMPT_NAME`, and :data:`ESCALATED_PROMPT_NAME` when the job escalated to the paid coder),
every rejected attempt's source and gate error through production's own
:class:`~noctis.research.failed_store.FailedAttemptStore` (pointed inside the workdir, so the
failure-census reader in :mod:`noctis.eval.failed_attempts` reads a bench exactly as it reads a
run), the validated file on success, and :data:`JOB_RECORD_NAME` — the same document the runner
keeps as the attempt's output, so the workdir is self-describing on its own.

**The knob layering, split by what a knob binds.** ``retries``, the output ceiling and the
escalation cap bind the *job*, so they are resolved per job from the request's own harness
(:func:`resolve_coder_knobs`: site defaults → config → bench override). The model, its thinking dial
and the sampling levers bind the *client*, which is built once per bench, so they are resolved
where the client is built (:func:`coder_clients`). Both halves come out of the same
:class:`~noctis.eval.coder_distill_sites.CoderKnobs` declaration, which is what makes an override
refusable before a bench spends anything.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

# The engine's own coder-client table (#345): a bench builds the clients production builds,
# through the composition root that owns them. Legal by the one-way rule — eval imports the
# engine, never the reverse (:mod:`noctis.eval.guard`).
from noctis.bootstrap import CoderClients, build_coder_clients
from noctis.eval.case import Case
from noctis.eval.coder_case import CoderPayload, coder_payload
from noctis.eval.coder_detectors import DegenerateFinding, inspect_strategy
from noctis.eval.coder_distill_sites import AuthoringJob, CoderKnobs, render_coder_prompt
from noctis.eval.harness import AsShipped, HarnessSpec, WorkedExample
from noctis.eval.metrics import AttemptOutcome
from noctis.eval.runner import Attempt, AttemptFn, AttemptRequest
from noctis.research.author import AS_SHIPPED as AUTHOR_AS_SHIPPED
from noctis.research.author import (
    AuthoringError,
    StrategyAuthor,
)
from noctis.research.author import WorkedExample as AuthorWorkedExample
from noctis.research.failed_store import FailedAttemptStore
from noctis.research.llm import Capabilities, Turn
from noctis.research.pricing import USAGE_FIELDS
from noctis.strategies import library
from noctis.strategies.families import FamilyRegistry
from noctis.strategies.scenario_spec import describe_spec

__all__ = [
    "ESCALATED_PROMPT_NAME",
    "FAILED_DIRNAME",
    "JOB_RECORD_NAME",
    "LIBRARY_DIRNAME",
    "NO_WORKING_LIBRARY",
    "PROMPT_NAME",
    "AttemptRecord",
    "CoderClients",
    "JobRecord",
    "authoring_engine",
    "coder_attempt",
    "coder_clients",
    "coder_input",
    "coder_site_input",
    "job_library",
    "job_record",
    "resolve_coder_knobs",
    "run_authoring_job",
    "strategy_name",
]

#: The library root a job builds inside its workdir — the two writable tiers hang off it.
LIBRARY_DIRNAME = "strategies"

#: Where production keeps rejected attempts under the working tier, quoted so a benched failure
#: folder is the same shape :mod:`noctis.eval.failed_attempts` already reads.
FAILED_DIRNAME = "failed"

#: The exact bytes the authoring engine sent, retained per job (and per escalated job).
PROMPT_NAME = "authoring-prompt.txt"
ESCALATED_PROMPT_NAME = "authoring-prompt-escalated.txt"

#: The job's own index over what it did — the same document the runner retains as the output.
JOB_RECORD_NAME = "job-record.json"

#: The working tiers a *preview* render points at: a path no checkout has, so the rendered prompt
#: sees the committed seeds and an empty working library, exactly as a fresh job's does. Never
#: created, never written — reads of a directory that is not there simply find nothing.
NO_WORKING_LIBRARY = "/nonexistent/noctis-eval/coder-preview"

# A strategy name is lower_snake_case (``library.NAME_RE``); a case id is a file stem that may
# carry hyphens and dots. Everything outside the alphabet folds to one underscore.
_UNSAFE = re.compile(r"[^a-z0-9]+")

# What a derived name is prefixed with when the case id does not start with a letter, so the
# derivation is total rather than occasionally refused by the gate's name rule.
_NAME_PREFIX = "case_"


# ── identity and the throwaway library ────────────────────────────────────────────────────


def strategy_name(case_id: str) -> str:
    """The strategy name one case authors under — derived from the case id, always gate-valid.

    Derived rather than declared because a coder case *is* a brief and production briefs carry no
    file name (the driver names the file); deriving it from the case id keeps the artifact
    traceable to the case that produced it and makes two reps of one case land the same name in
    two different libraries.
    """
    token = _UNSAFE.sub("_", (case_id or "").strip().lower()).strip("_")
    if not token or not token[0].isalpha():
        token = f"{_NAME_PREFIX}{token}".rstrip("_")
    return token


def job_library(workdir: Path | str, *, seeds: Path | str) -> library.LibraryPaths:
    """One job's three tiers: the committed seeds in place, fresh working tiers under ``workdir``.

    The seeds tier is referenced, never copied — see the module docstring for why that is safe and
    why copying would be worse. The two writable tiers are created empty here, so a job that never
    lands a file still leaves a library-shaped directory a reader can look into.
    """
    root = Path(workdir) / LIBRARY_DIRNAME
    paths = library.LibraryPaths(
        seeds=Path(seeds),
        tmp=root / library.TMP_SUBDIR,
        champions=root / library.CHAMPIONS_SUBDIR,
    )
    paths.tmp.mkdir(parents=True, exist_ok=True)
    paths.champions.mkdir(parents=True, exist_ok=True)
    return paths


# ── the knobs, layered ────────────────────────────────────────────────────────────────────


def resolve_coder_knobs(settings: Any, harness: HarnessSpec = HarnessSpec()) -> CoderKnobs:
    """The coder's knob set for one bench: site defaults → config → bench override.

    The site's declaration supplies the defaults, the resolved settings supply what an operator
    configured, and the harness spec's overrides sit on top — the layering the knob module names,
    performed in one place. An override naming a knob the site does not declare never reaches here:
    :meth:`~noctis.eval.harness.HarnessSpec.validate_knobs` refuses it during the preflight.
    """
    agent = getattr(getattr(settings, "research", None), "agent", None)
    values: dict[str, Any] = {}
    for name in sorted(CoderKnobs.knob_names()):
        if agent is not None and hasattr(agent, name):
            values[name] = getattr(agent, name)
    values.update({str(name): value for name, value in harness.knob_overrides.items()})
    return CoderKnobs(**values)


def _worked_example(selection: WorkedExample) -> AuthorWorkedExample:
    """The harness's worked-example selection as the engine's own three-valued dial.

    The two ``AS_SHIPPED`` sentinels are deliberate twins rather than one shared object (the engine
    may never import this layer), so the crossing is made here, once, by name.
    """
    if isinstance(selection, AsShipped):
        return AUTHOR_AS_SHIPPED
    return selection


def authoring_engine(
    *,
    client: Any,
    strategies_dir: library.LibrarySpec,
    families: FamilyRegistry,
    harness: HarnessSpec = HarnessSpec(),
    knobs: CoderKnobs = CoderKnobs(),
    attempt_timeout: float | None = None,
) -> StrategyAuthor:
    """One :class:`~noctis.research.author.StrategyAuthor` under a benchmark's harness spec.

    **The mapping this story owns**, in one function: the five ablation dials become the engine's
    five composition parameters, and the two job-binding knobs become its output ceiling and its
    private-retry budget. A knob left ``None`` is *not passed at all*, so the engine's own built-in
    default stands — an unstated knob must never become a benchmark stating one.
    """
    kwargs: dict[str, Any] = {}
    if knobs.coder_max_tokens is not None:
        kwargs["max_tokens"] = int(knobs.coder_max_tokens)
    if knobs.coder_retries is not None:
        kwargs["retries"] = int(knobs.coder_retries)
    return StrategyAuthor(
        client=client,
        strategies_dir=strategies_dir,
        families=families,
        include_contract_sheet=harness.contract_sheet,
        include_template=harness.template,
        worked_example=_worked_example(harness.worked_example),
        include_feasibility_rules=harness.feasibility_rules,
        include_retry_hints=harness.retry_hints,
        attempt_timeout=attempt_timeout,
        **kwargs,
    )


# ── the records ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AttemptRecord:
    """One *internal* coder attempt: what it cost, what answered it, and how the gate ruled.

    ``attempt`` is the engine's own 1-based index (1 = the first ask, 2… = a private retry) and
    restarts at 1 for an escalated job, which is why ``escalated`` rides beside it rather than
    being inferred from the index. ``artifact`` is where the bytes went, relative to the job's
    workdir — the retained failure record for a rejection, the validated file for the attempt that
    landed — so a reader gets from a record to the source without a search.
    """

    attempt: int
    passed: bool
    escalated: bool = False
    error: str | None = None
    model: str | None = None
    served_model: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    artifact: str | None = None

    def document(self) -> dict[str, Any]:
        """This attempt as the JSON a record carries — every field, absent ones as ``null``."""
        return {
            "attempt": self.attempt,
            "passed": self.passed,
            "escalated": self.escalated,
            "error": self.error,
            "model": self.model,
            "served_model": self.served_model,
            "usage": dict(self.usage),
            "artifact": self.artifact,
        }


@dataclass(frozen=True)
class JobRecord:
    """One whole authoring job: the gate's verdict, and every internal attempt behind it.

    The runner keeps this as the attempt's output, and the job keeps a copy beside its own
    artifacts, so the same document answers "what happened here?" from either end.

    ``findings`` is what the degenerate-pass detectors said about the file this job landed — empty
    for a job that landed none, and empty for a clean pass. They ride *beside* the verdict and
    never in it: :attr:`passed` is the gate's, and nothing here consults a finding.
    """

    case_id: str
    strategy: str
    passed: bool
    attempts: tuple[AttemptRecord, ...] = ()
    error: str | None = None
    path: str | None = None
    model: str | None = None
    seconds: float | None = None
    findings: tuple[DegenerateFinding, ...] = ()

    @property
    def escalated(self) -> bool:
        """Whether this job fell back to the paid coder at any point."""
        return any(one.escalated for one in self.attempts)

    @property
    def first_attempt_passed(self) -> bool | None:
        """Whether the opening ask landed a file, or ``None`` for a job that never asked."""
        return self.attempts[0].passed if self.attempts else None

    def usage(self) -> dict[str, int]:
        """What the whole job spent — a field summed only when every attempt reported it.

        All-or-nothing per field, the same posture :class:`~noctis.eval.metrics.AttemptOutcome`
        takes: a partial sum would price a job that is really unpriceable.
        """
        if not self.attempts:
            return {}
        spent: dict[str, int] = {}
        for name in USAGE_FIELDS:
            if all(name in one.usage for one in self.attempts):
                spent[name] = sum(int(one.usage[name]) for one in self.attempts)
        return spent

    def document(self) -> dict[str, Any]:
        """The job as one JSON object — the artifact both the runner and the workdir carry."""
        return {
            "case_id": self.case_id,
            "strategy": self.strategy,
            "passed": self.passed,
            "escalated": self.escalated,
            "first_attempt_passed": self.first_attempt_passed,
            "error": self.error,
            "path": self.path,
            "model": self.model,
            "seconds": self.seconds,
            "attempts": [one.document() for one in self.attempts],
            "findings": [
                {
                    "detector": one.detector,
                    "severity": one.severity,
                    "summary": one.summary,
                }
                for one in self.findings
            ],
        }

    def attempt(self) -> Attempt:
        """This job as the one runner attempt it is — the two-level design's crossing point.

        The runner sees a single scored attempt whose verdict is the gate's; the per-attempt
        records ride in the output, which is what the runner retains beside the case and what a
        site scorer is handed as this job's reply.
        """
        answered = [one.served_model for one in self.attempts if one.served_model]
        return Attempt(
            outcome=AttemptOutcome(
                passed=self.passed,
                model=self.model,
                seconds=self.seconds,
                error=self.error,
                **self.usage(),
            ),
            output=json.dumps(self.document(), sort_keys=True, default=str),
            served_model=answered[-1] if answered else None,
        )


def job_record(document: Mapping[str, Any]) -> JobRecord:
    """One retained job document read back — the reader half of :meth:`JobRecord.document`."""
    return JobRecord(
        case_id=str(document.get("case_id") or ""),
        strategy=str(document.get("strategy") or ""),
        passed=bool(document.get("passed")),
        attempts=tuple(_attempt_record(entry) for entry in document.get("attempts") or ()),
        error=_text(document.get("error")),
        path=_text(document.get("path")),
        model=_text(document.get("model")),
        seconds=None if document.get("seconds") is None else float(document["seconds"]),
        findings=tuple(_finding(entry) for entry in document.get("findings") or ()),
    )


def _finding(entry: Mapping[str, Any]) -> DegenerateFinding:
    """One retained detector finding back as the advisory record it is."""
    return DegenerateFinding(
        detector=str(entry.get("detector") or ""),
        severity=str(entry.get("severity") or ""),
        summary=str(entry.get("summary") or ""),
    )


def _attempt_record(entry: Mapping[str, Any]) -> AttemptRecord:
    """One retained attempt row back as its typed record."""
    return AttemptRecord(
        attempt=int(entry.get("attempt") or 0),
        passed=bool(entry.get("passed")),
        escalated=bool(entry.get("escalated")),
        error=_text(entry.get("error")),
        model=_text(entry.get("model")),
        served_model=_text(entry.get("served_model")),
        usage={str(key): int(value) for key, value in (entry.get("usage") or {}).items()},
        artifact=_text(entry.get("artifact")),
    )


def _text(value: Any) -> str | None:
    """A retained string field, or ``None`` — absence stays absence, never an empty string."""
    return None if value is None else str(value)


# ── the client, observed ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Observed:
    """What one completion reported: the billed usage it recorded, and the model that served it."""

    usage: Mapping[str, int] = field(default_factory=dict)
    served_model: str | None = None


class _RecordingClient:
    """The coder client, wrapped so each completion's usage and served model survive the job.

    The engine's per-attempt seam reports the attempt number, the gate error and the source — it
    does not report what the transport billed, and it should not: the engine holds no accounting.
    So the accounting stays out here, exactly as the production toolbox keeps its own call counter
    out here, and the two are joined by the one fact that makes them joinable — **attempt ``k`` is
    completion ``k``**, since every attempt makes exactly one completion. Observations are keyed by
    that number rather than appended, so a completion that timed out (and may answer long after the
    attempt was abandoned) can never be read as the next attempt's.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self.capabilities = client.capabilities
        self.calls = 0
        self.observed: dict[int, _Observed] = {}

    def __getattr__(self, name: str) -> Any:
        """Anything else the engine duck-types off a client (``thinking_enabled``) is forwarded."""
        return getattr(self._client, name)

    def complete(self, **kwargs: Any) -> Turn:
        self.calls += 1
        number = self.calls
        turn = self._client.complete(**kwargs)
        self.observed[number] = _Observed(
            usage=_billed(turn.usage), served_model=turn.served_model or None
        )
        return turn


def _billed(usage: Mapping[str, Any] | None) -> dict[str, int]:
    """The four billed fields a completion really reported — an absent one stays absent."""
    return {
        name: int(count)
        for name, count in (usage or {}).items()
        if name in USAGE_FIELDS and count is not None
    }


class _PreviewClient:
    """A client for rendering only: it carries capabilities and refuses to complete anything.

    The site's declared renderer needs an engine, and an engine needs a client — but a *preview*
    (the prompt the runner retains beside a case) asks nothing. Refusing rather than returning an
    empty turn keeps that a fact about the object instead of a promise about the control flow.
    """

    capabilities = Capabilities()

    def complete(self, **kwargs: Any) -> Turn:  # pragma: no cover - refuses by construction
        raise RuntimeError(
            "the coder site's preview engine renders prompts and never completes one — a benched "
            "authoring job runs through coder_attempt(), which is handed a real client"
        )


# ── the ask: one case as the site's renderer input ────────────────────────────────────────


def coder_input(
    case: Case, *, seeds: Path | str, harness: HarnessSpec = HarnessSpec()
) -> AuthoringJob:
    """One coder case as the :class:`~noctis.eval.coder_distill_sites.AuthoringJob` a site renders.

    The engine here is a *preview* engine: the committed seeds (so the prompt carries the shipped
    template and worked example a real install folds in) over an empty working library at
    :data:`NO_WORKING_LIBRARY`, so the rendered prompt sees exactly what a fresh job's library holds
    and never an operator's own drafts.
    """
    payload = coder_payload(case)
    paths = library.LibraryPaths(
        seeds=Path(seeds),
        tmp=Path(NO_WORKING_LIBRARY) / library.TMP_SUBDIR,
        champions=Path(NO_WORKING_LIBRARY) / library.CHAMPIONS_SUBDIR,
    )
    engine = authoring_engine(
        client=_PreviewClient(),
        strategies_dir=paths,
        families=FamilyRegistry(),
        harness=harness,
    )
    return _job_for(engine, case.case_id, payload)


def coder_site_input(settings: Any) -> Callable[[Case], AuthoringJob]:
    """The adapter a bench hands the runner as its ``site_input`` for the coder site."""
    seeds = Path(settings.strategies_dir)

    def adapt(case: Case) -> AuthoringJob:
        return coder_input(case, seeds=seeds)

    return adapt


def _job_for(engine: StrategyAuthor, case_id: str, payload: CoderPayload) -> AuthoringJob:
    """One engine plus one case as the authoring job the site's renderer takes."""
    return AuthoringJob(
        author=engine,
        name=strategy_name(case_id),
        brief=payload.brief,
        spec=payload.scenario_spec,
    )


# ── the clients a live coder bench asks through ───────────────────────────────────────────


def coder_clients(
    settings: Any,
    *,
    model: str | None = None,
    client: Any = None,
    knobs: CoderKnobs | None = None,
) -> CoderClients:
    """The clients one live coder bench runs on — production's dials, the bench's model on top.

    Built by the composition root's own :func:`~noctis.bootstrap.build_coder_clients`, from this
    bench's resolved knob set: a bench that assembled its own clients would be measuring a
    configuration nobody ships, and the coder's dials (``deliberate=True``, the two thinking dials,
    the sampling levers) live in one table on the engine's side of the one-way boundary.

    Two contracts of the bench's own sit on top of the engine's. **A bench refuses where a session
    degrades**: a research session with no coder client assembles in driver-authored mode, but a
    coder bench with no coder has nothing to measure, so an unbuildable model is
    :class:`~noctis.eval.bootstrap.LiveModelUnavailable` rather than a quiet ``None``. And
    **``client`` is the seam a test injects**: when it is given nothing is built and no fallback is
    invented.
    """
    from noctis.eval.bootstrap import LiveModelUnavailable
    from noctis.research.llm import resolved_research_model

    resolved = knobs if knobs is not None else resolve_coder_knobs(settings)
    alias = model or resolved.coder_model or resolved_research_model(settings)
    if client is not None:
        return CoderClients(client=client, model=alias)
    # The escalation cap is spent per *bench*, which has no session to count in, so it gates the
    # paid client at build time: a bench that will never escalate never even resolves the provider.
    built = build_coder_clients(
        settings, knobs=resolved, model=alias, escalations=int(resolved.max_escalations or 0)
    )
    if built.client is None:
        raise LiveModelUnavailable(
            f"no coder client for {alias!r} — the coder site authors through a real model client, "
            "so its provider key and the [llm] extra have to be present"
        )
    return built


# ── one authoring job ─────────────────────────────────────────────────────────────────────


def coder_attempt(
    settings: Any,
    *,
    client: Any,
    model: str | None = None,
    fallback_client: Any = None,
    fallback_model: str | None = None,
    attempt_timeout: float | None = None,
    seeds: Path | str | None = None,
) -> AttemptFn:
    """The attempt callable that authors one benched case per call — the runner's coder seam.

    One runner attempt is one whole authoring job (see the module docstring). The knobs that bind
    the job are resolved per request from that request's own harness, so a bench spec's override
    reaches the engine; the clients are built once by the caller, because that is where the knobs
    that bind a client were already resolved.
    """
    library_seeds = Path(seeds) if seeds is not None else Path(settings.strategies_dir)

    def attempt(request: AttemptRequest) -> Attempt:
        record = run_authoring_job(
            request,
            client=client,
            seeds=library_seeds,
            knobs=resolve_coder_knobs(settings, request.harness),
            model=model,
            fallback_client=fallback_client,
            fallback_model=fallback_model,
            attempt_timeout=attempt_timeout,
        )
        return record.attempt()

    return attempt


def run_authoring_job(
    request: AttemptRequest,
    *,
    client: Any,
    seeds: Path | str,
    knobs: CoderKnobs = CoderKnobs(),
    model: str | None = None,
    fallback_client: Any = None,
    fallback_model: str | None = None,
    attempt_timeout: float | None = None,
) -> JobRecord:
    """Author one benched case in its own throwaway library, and record everything it did.

    The gate is the verdict: a validated file means the job passed, and an
    :class:`~noctis.research.author.AuthoringError` (or a brief the engine refuses before spending a
    completion — an unknown ``reference``) means it failed, carrying that error verbatim. When the
    local coder spends its whole retry budget and the escalation cap allows it, the *same* brief
    escalates to the paid fallback with its own full budget, exactly as a research session
    escalates — and every attempt on both sides is recorded.
    """
    payload = coder_payload(request.case)
    name = strategy_name(request.case.case_id)
    workdir = Path(request.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    paths = job_library(workdir, seeds=seeds)
    families = FamilyRegistry()
    store = FailedAttemptStore(paths.tmp / FAILED_DIRNAME)
    oracle = describe_spec(payload.scenario_spec) if payload.scenario_spec is not None else None

    records: list[AttemptRecord] = []
    started = time.perf_counter()
    result, failure = _author(
        client=client,
        request=request,
        payload=payload,
        name=name,
        paths=paths,
        families=families,
        knobs=knobs,
        model=model,
        attempt_timeout=attempt_timeout,
        store=store,
        oracle=oracle,
        workdir=workdir,
        records=records,
        prompt_name=PROMPT_NAME,
        escalated=False,
    )
    if _may_escalate(failure, fallback_client, knobs):
        result, failure = _author(
            client=fallback_client,
            request=request,
            payload=payload,
            name=name,
            paths=paths,
            families=families,
            knobs=knobs,
            model=fallback_model,
            attempt_timeout=attempt_timeout,
            store=store,
            oracle=oracle,
            workdir=workdir,
            records=records,
            prompt_name=ESCALATED_PROMPT_NAME,
            escalated=True,
        )
    seconds = round(time.perf_counter() - started, 6)

    landed = _landed(result, workdir)
    if landed is not None and records:
        records[-1] = replace(records[-1], artifact=landed)
    record = JobRecord(
        case_id=request.case.case_id,
        strategy=name,
        passed=result is not None,
        attempts=tuple(records),
        error=None if failure is None else _gate_error(failure),
        path=landed,
        model=_model_of(records, model),
        seconds=seconds,
        findings=_findings(name, families) if result is not None else (),
    )
    _keep(workdir / JOB_RECORD_NAME, json.dumps(record.document(), indent=2, default=str) + "\n")
    return record


def _findings(name: str, families: FamilyRegistry) -> tuple[DegenerateFinding, ...]:
    """What the degenerate-pass detectors say about the file this job just landed.

    The class is the one the gate itself installed — ``write_strategy`` registers the validated
    file into this job's own registry — so the detectors read exactly the artifact that passed,
    with no second import and no path guessing. Nothing is caught here on purpose: every operation
    :func:`~noctis.eval.coder_detectors.inspect_strategy` performs (``param_space()``,
    ``params_cls()``, a replay over ``fixture_frame()``) is one the gate already performed on this
    very file, so a raise would be a real defect rather than weather to swallow.
    """
    if name not in families:
        return ()  # pragma: no cover - a landed file is registered by the gate that landed it
    return inspect_strategy(families.get_class(name)).findings


def _may_escalate(failure: Exception | None, fallback_client: Any, knobs: CoderKnobs) -> bool:
    """Whether a failed local author may fall back to the paid coder — production's own rule.

    Only a **spent retry budget** escalates (:class:`~noctis.research.author.AuthoringError`): a
    brief the engine refused before spending a completion — an unknown ``reference`` — is a bad
    brief, and paying a strong model to be told the same thing would buy nothing. Beside that, the
    two conditions a session applies: a fallback client exists, and the escalation cap is not zero.
    """
    return (
        isinstance(failure, AuthoringError)
        and fallback_client is not None
        and int(knobs.max_escalations or 0) > 0
    )


def _author(
    *,
    client: Any,
    request: AttemptRequest,
    payload: CoderPayload,
    name: str,
    paths: library.LibraryPaths,
    families: FamilyRegistry,
    knobs: CoderKnobs,
    model: str | None,
    attempt_timeout: float | None,
    store: FailedAttemptStore,
    oracle: str | None,
    workdir: Path,
    records: list[AttemptRecord],
    prompt_name: str,
    escalated: bool,
) -> tuple[dict[str, Any] | None, Exception | None]:
    """One engine's whole run at the brief: the file it landed, or the error it ended on.

    Both halves of the return are meaningful and exactly one is ``None`` — a job either has a
    validated file or a retained gate error, never both and never neither.
    """
    recorder = _RecordingClient(client)
    engine = authoring_engine(
        client=recorder,
        strategies_dir=paths,
        families=families,
        harness=request.harness,
        knobs=knobs,
        attempt_timeout=attempt_timeout,
    )
    # The bytes this engine is about to send, retained beside the job. The renderer is total —
    # a brief production refuses before composing anything renders its refusal — so a case the
    # corpus deliberately ships to be refused keeps an artifact rather than raising here.
    _keep(
        workdir / prompt_name,
        render_coder_prompt(_job_for(engine, request.case.case_id, payload), request.harness),
    )

    def sink(attempt: int, error: Exception | None, source: str) -> None:
        artifact: str | None = None
        if error is not None:
            kept = store.record(name, attempt, source, str(error), oracle=oracle)
            artifact = str(kept.relative_to(workdir))
        observed = recorder.observed.get(attempt, _Observed())
        records.append(
            AttemptRecord(
                attempt=attempt,
                passed=error is None,
                escalated=escalated,
                error=None if error is None else str(error),
                model=model,
                served_model=observed.served_model,
                usage=observed.usage,
                artifact=artifact,
            )
        )

    try:
        return engine.author(name, payload.brief, on_attempt=sink, spec=payload.scenario_spec), None
    except (AuthoringError, library.StrategyValidationError) as refusal:
        return None, refusal


def _landed(result: Mapping[str, Any] | None, workdir: Path) -> str | None:
    """Where the validated file landed, relative to the workdir — ``None`` when none did."""
    if result is None:
        return None
    path = Path(str(result.get("path") or ""))
    try:
        return str(path.relative_to(workdir))
    except ValueError:  # pragma: no cover - a job always writes inside its own workdir
        return str(path)


def _gate_error(failure: Exception) -> str:
    """The gate's own words for a failed job — an authoring error carries its final one."""
    if isinstance(failure, AuthoringError) and failure.validation_error is not None:
        return str(failure.validation_error)
    return str(failure)


def _model_of(records: Sequence[AttemptRecord], model: str | None) -> str | None:
    """The model that made the job's last attempt, or the one it was asked to start with."""
    for record in reversed(records):
        if record.model:
            return record.model
    return model


def _keep(target: Path, text: str) -> None:
    """Retain one artifact inside the job's own workdir."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
