"""The bench runner — the eval layer's one impure module: a directory, a clock, and no more.

Everything else under :mod:`noctis.eval` is a pure function of values (a corpus deals its own split,
the metrics are arithmetic, the record is a builder beside a validator). This module is where a
benchmark meets the world, so it is deliberately the *only* place in the layer that opens a file or
reads a clock — and even here the model call is not made: it arrives as an injected
:data:`AttemptFn`, so the whole suite runs a full bench end to end without a network.

**A bench run is not a run, and the layout says so.** It mints its own id (the run-id minter's
shape and discipline, by identity — :func:`new_bench_id`) and owns exactly one directory:
``<workspace>/bench/<bench_id>/``. The bench area hangs off the **workspace root**, beside the
shared ``data_lake/`` and for the same reason: a benchmark is run-neutral, and nothing about it
belongs to the run that happened to be open when somebody started it. Nothing under ``runs/`` is
read or written, no run lock is taken, no champions board is touched, no run-board entry appears —
the runner is handed its bench root and knows no other path. :data:`BENCH_INDEX_NAME` names the
slot a derived listing roll-up will take later; this story leaves it empty rather than guessing at
a shape, because an index is a *derived* view and deriving one over a single bench is not a design.

**The evidence is retained per case, and replayable from the tree alone.** Each job — one case, one
model configuration, one rep — gets ``cases/<case_id>/<config_id>/rep-NN/``, holding the rendered
prompt it was asked with, each attempt's output and error, and a ``job.json`` index over them.
Inside it sits ``work/``: the throwaway area the attempt callable is handed and confined to, fresh
and empty per job, so nothing one case writes can be seen (or overwritten) by the next.
:func:`replay` rebuilds all of it from the directory, which is what "replayable" has to mean — a
bench directory copied to another machine still answers what every case was asked and what it said.
Artifacts land as each job finishes, so a bench killed halfway keeps everything it already learned.

**Nothing is spent before it is stated.** :meth:`BenchRunner.preflight` is the dry run: it resolves
the site, refuses a harness spec the site cannot accept, counts the jobs, and prices them from the
pricing table and a caller-supplied per-attempt token estimate — calling no attempt and writing
nothing at all. It is also the *only* code path that computes a plan, so the dry run and the live
run can never drift apart. :meth:`BenchRunner.run` then refuses to start unless it is handed that
plan back: a run whose shape differs from the acknowledged one is refused by name, because
acknowledging two cases must not authorize two hundred. The estimate inherits
:mod:`noctis.research.pricing`'s posture whole — a model the table cannot price yields ``None``
(rendered ``n/a``), never a confident zero.

**The site scores its own bench, and the runner names no site.** Once every job has answered, the
runner hands the whole set of :class:`~noctis.eval.site.AnsweredCase`\\ s to each scorer the *site's
declaration* carries and folds what comes back into ``harness.dials`` — the one subtree a record
quotes verbatim. That is the whole integration: a site with no scorers writes exactly the record it
wrote before scoring existed, a site with one publishes its reading beside the composed dials, and
neither this module nor the ``bench`` verbs contain a line of any site's vocabulary. A reading that
would overwrite a dial the harness composed is refused (:class:`ConflictingReading`) rather than
merged, because a record whose ``is_production`` came from a scorer is a record that lies about what
was asked.

**The record is the pure builder's, or there is no record.** The runner collects
:class:`~noctis.eval.record.BenchArtifacts`, hands them to :func:`~noctis.eval.record.build`, checks
the document with :func:`~noctis.eval.record.validate`, and writes ``bench.json`` atomically only if
it validates clean. An invalid document is refused, not written — the artifacts it was derived from
stay on disk, because they are what an operator debugs the refusal with.

**How the jobs are worked is a seam, not a fact about a bench.** :class:`SequentialExecutor` runs
each job in order in this process and declares its one worker;
:class:`~noctis.eval.pool.PooledExecutor` works them on a pool of forked workers and survives the
pool's own failure modes. Both are :class:`JobExecutor`s, so widening a bench to a pool changed
what is injected here rather than anything this module guarantees — the record a pooled bench
writes is the record the sequential one writes.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar

from noctis.eval.case import Case, Split
from noctis.eval.corpus import Corpus
from noctis.eval.harness import AS_SHIPPED, AsShipped, HarnessSpec, WorkedExample
from noctis.eval.identity import SiteIdentity, site_identity
from noctis.eval.metrics import AttemptOutcome, CaseResult
from noctis.eval.record import (
    BenchArtifacts,
    CaseRun,
    CorpusIdentity,
    EngineStamp,
    ModelConfig,
    build,
    validate,
)
from noctis.eval.registry import site
from noctis.eval.site import AgentSite, AnsweredCase
from noctis.eval.taxonomy import FailureTaxonomy
from noctis.observability.debug.runid import new_run_id
from noctis.observability.engine_id import fingerprint
from noctis.research.pricing import PriceTable, default_table

__all__ = [
    "BENCH_DIRNAME",
    "BENCH_INDEX_NAME",
    "BENCH_RECORD_NAME",
    "CASES_DIRNAME",
    "DEFAULT_CONFIGS",
    "JOB_NAME",
    "PROMPT_NAME",
    "UNSTATED_MODEL",
    "WHOLE_CORPUS",
    "WORK_DIRNAME",
    "Attempt",
    "AttemptArtifact",
    "AttemptFn",
    "BenchJob",
    "BenchRun",
    "BenchRunner",
    "CaseArtifacts",
    "ConflictingReading",
    "InvalidBenchRecord",
    "JobExecutor",
    "SequentialExecutor",
    "SpendPlan",
    "SpendUnacknowledged",
    "UnsafeArtifactName",
    "WorkedJob",
    "bench_root",
    "engine_stamp",
    "harness_dials",
    "harness_hash",
    "new_bench_id",
    "replay",
]

# ── the layout, named once ────────────────────────────────────────────────────────────────

#: The workspace-level bench area — run-neutral, exactly like ``data_lake/``.
BENCH_DIRNAME = "bench"

#: The one document a bench run publishes, beside the per-case evidence it was derived from.
BENCH_RECORD_NAME = "bench.json"

#: The slot a derived listing roll-up over every bench record will take (plan 08's). Named here so
#: the layout is stated in one place; deliberately never written by this story — an index is a
#: derived view, and a derived view nobody has designed is better absent than guessed.
BENCH_INDEX_NAME = "index.json"

CASES_DIRNAME = "cases"
WORK_DIRNAME = "work"
PROMPT_NAME = "prompt.txt"
JOB_NAME = "job.json"

#: What ``corpus.split`` says on a bench that measured the whole corpus. A word rather than
#: ``null``: "both halves" and "nobody recorded which half" are very different statements.
WHOLE_CORPUS = "all"

#: How a configuration that never named a model is spelled in a preflight's unpriced list. It is
#: unpriceable for the same reason an unknown model is, and saying so beats an empty string.
UNSTATED_MODEL = "unknown"

#: The configuration a bench runs under when the caller names none — one slot, honestly unstated,
#: so the record's provenance tier reports "nobody told the runner which model" rather than nothing.
DEFAULT_CONFIGS: tuple[ModelConfig, ...] = (ModelConfig(config_id="default"),)

# The harness hash's rendering scheme, hashed in as a prefix: a change to *how* a spec is rendered
# (not to what it composes) is a new scheme name, so an old hash is never silently re-meant.
_HARNESS_SCHEME = "harness/1"

# A truncated SHA-256 — the width every other identity in this repo prints.
_DIGEST_CHARS = 16

_STAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class SpendUnacknowledged(RuntimeError):
    """A live bench asked to start on a plan nobody ran the preflight for, or on a different one."""


class InvalidBenchRecord(ValueError):
    """The collected artifacts do not build a readable record. Its message names every problem."""


class UnsafeArtifactName(ValueError):
    """A case or configuration id that would not stay inside its own directory."""


class ConflictingReading(ValueError):
    """A site's scorer published a dial the harness composition already publishes.

    Refused rather than merged in either direction: the composed dials say what the model was
    asked, a reading says what its answers earned, and a record in which one silently overwrote the
    other would misdescribe the very thing it exists to make comparable.
    """


# ── identity and the bench area ───────────────────────────────────────────────────────────


def new_bench_id(now: datetime | None = None) -> str:
    """Mint a fresh bench id — minted, never derived, in the run id's own shape.

    The minter is :func:`~noctis.observability.debug.runid.new_run_id` by identity rather than by a
    second implementation: sortable by name, collision-free across concurrent benches, greppable.
    A bench id is not a run id and never addresses one — what tells them apart is the tree they
    live in, which is the whole point of the bench area hanging off the workspace root.
    """
    return new_run_id(now)


def bench_root(workspace_dir: Path | str) -> Path:
    """The bench area of one workspace: ``<workspace>/bench``.

    Workspace-level, beside the shared data lake, and deliberately **not** one of the run-scoped
    derived paths (``state_dir``, ``reports_dir``, ``qa_dir``, ``memory_path``): a benchmark
    measures the engine, not one run's trajectory, so two runs in a workspace read one bench area
    the same way they read one lake.
    """
    return Path(workspace_dir) / BENCH_DIRNAME


def engine_stamp(root: Path | None = None) -> EngineStamp:
    """This checkout's engine identity, as the record's stamp: declared version plus digests."""
    computed = fingerprint(root)
    return EngineStamp(
        engine_version=computed.engine_version,
        fingerprint=computed.digests(),
        noctis_version=_noctis_version(),
    )


# ── the harness spec, hashed ──────────────────────────────────────────────────────────────


def harness_dials(spec: HarnessSpec) -> dict[str, object]:
    """One harness spec as the dials a record quotes verbatim — the ablation, as data.

    ``worked_example`` is rendered with its selection *tagged* (``as_shipped``, ``none``,
    ``named:<name>``) because the three are different compositions and a strategy that happened to
    be called ``as_shipped`` must not read as the shipped example. ``is_production`` rides along so
    a reader can see at a glance whether the bench deviated from production at all.
    """
    return {
        "contract_sheet": spec.contract_sheet,
        "template": spec.template,
        "worked_example": _worked_example(spec.worked_example),
        "feasibility_rules": spec.feasibility_rules,
        "retry_hints": spec.retry_hints,
        "knob_overrides": {str(name): value for name, value in sorted(spec.knob_overrides.items())},
        "is_production": spec.is_production,
    }


def harness_hash(spec: HarnessSpec) -> str:
    """The content hash of one prompt composition — a key component of every bench record.

    Taken over exactly the dials :func:`harness_dials` publishes, so a record can never carry a hash
    of something other than the composition it shows. Deterministic by construction: canonical JSON
    with sorted keys, so the same spec hashes identically in any process.
    """
    body = json.dumps(
        {"scheme": _HARNESS_SCHEME, "dials": harness_dials(spec)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]


def _worked_example(selection: WorkedExample) -> str:
    """The worked-example selection as one tagged token — see :func:`harness_dials`."""
    if isinstance(selection, AsShipped):
        return AS_SHIPPED.value
    return "none" if selection is None else f"named:{selection}"


# ── the model call: the one seam this module never implements ─────────────────────────────


@dataclass(frozen=True)
class AttemptRequest:
    """One ask, as the attempt callable receives it: the prompt, its case, and where to work.

    A single frozen request rather than a widening argument list, so the day an attempt needs one
    more fact (a knob set, a deadline) is a new field here instead of a broken signature at every
    call site. ``workdir`` is this job's throwaway area — fresh, empty, and the only directory an
    attempt has any business writing into.
    """

    prompt: str
    case: Case
    config: ModelConfig
    harness: HarnessSpec
    rep: int
    attempt: int
    workdir: Path


@dataclass(frozen=True)
class Attempt:
    """What one model call produced: the scored outcome, the answer text, the model that served it.

    ``outcome`` is :mod:`noctis.eval.metrics`' own input type, untouched, so the record's aggregates
    are that module's arithmetic. ``output`` is retained as an artifact and never scored here — the
    site's oracle is the judge, and ``outcome.passed`` is its verdict.
    """

    outcome: AttemptOutcome
    output: str | None = None
    served_model: str | None = None


#: The injected model call. It answers one attempt; the runner decides whether to ask again.
AttemptFn = Callable[[AttemptRequest], Attempt]


# ── the execution seam ────────────────────────────────────────────────────────────────────

_Job = TypeVar("_Job")
_Done = TypeVar("_Done")


class JobExecutor(Protocol):
    """How a bench's jobs are worked through — one at a time here, or on a pool of workers.

    ``workers`` is the seam's own declaration of its width — one for the sequential executor, the
    configured count for the pool — so a record or a log can state how a bench was worked without
    asking what class it was.
    """

    @property
    def workers(self) -> int:
        """How many jobs this executor may have in flight at once."""

    def run(self, work: Callable[[_Job], _Done], jobs: Sequence[_Job]) -> tuple[_Done, ...]:
        """Apply ``work`` to every job, returning the results in job order."""


@dataclass(frozen=True)
class SequentialExecutor:
    """One job at a time, in this process, in order — the honest default.

    A bench's cost is dominated by a network round trip, so parallelism is worth having; it is also
    where a benchmark quietly stops being reproducible (interleaved writes, a shared working
    directory, an ordering-dependent rate limit). This is the executor that cannot get any of that
    wrong, and it stays the default for exactly that reason: a bench that wants width says so by
    injecting :class:`~noctis.eval.pool.PooledExecutor`.
    """

    workers: int = 1

    def run(self, work: Callable[[_Job], _Done], jobs: Sequence[_Job]) -> tuple[_Done, ...]:
        """Every job, worked in order — a job that raises is the caller's to handle."""
        return tuple(work(job) for job in jobs)


# ── the plan ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SpendPlan:
    """What a bench would run, and what that would cost — the dry run's whole output.

    It is also the token a live run must be handed back: two plans compare equal only when every
    field does, so acknowledging one shape can never authorize another. Every dollar figure is a
    **ceiling** (every job taking every attempt) and an **estimate** from a versioned table — the
    pricing module's posture, carried in the field name so nobody reads it as an invoice.
    """

    site_id: str
    corpus_digest: str
    split: str
    cases: int
    reps: int
    configs: tuple[str, ...]
    max_attempts: int
    jobs: int
    attempts_ceiling: int
    tokens_ceiling: int | None
    usd_ceiling_estimate: float | None
    pricing_table_version: str
    unpriced_models: tuple[str, ...] = ()

    def summary(self) -> str:
        """The one line an operator reads before agreeing to spend anything."""
        dollars = (
            "n/a"
            if self.usd_ceiling_estimate is None
            else f"${self.usd_ceiling_estimate:,.2f} (estimate)"
        )
        unpriced = f"; unpriced: {', '.join(self.unpriced_models)}" if self.unpriced_models else ""
        return (
            f"{self.site_id}/{self.split}: {self.cases} case(s) × {self.reps} rep(s) × "
            f"{len(self.configs)} config(s) = {self.jobs} job(s), up to "
            f"{self.attempts_ceiling} attempt(s); ceiling {dollars} under pricing table "
            f"{self.pricing_table_version}{unpriced}"
        )


# ── the per-case artifacts ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AttemptArtifact:
    """One attempt as it survives on disk: what it answered, and what went wrong."""

    index: int
    passed: bool
    output: str | None
    error: str | None


@dataclass(frozen=True)
class CaseArtifacts:
    """One job's retained evidence, read back from its own directory."""

    case_id: str
    config_id: str
    rep: int
    directory: Path
    prompt: str
    attempts: tuple[AttemptArtifact, ...]


def replay(bench_dir: Path | str) -> tuple[CaseArtifacts, ...]:
    """Every completed job's evidence, rebuilt from a bench directory alone.

    Keyed off each job's own ``job.json``, which is written **after** its attempts finish — so a
    bench still running (or one that was killed) replays exactly the jobs that completed, never a
    half-written one. Ordered by case, configuration and rep, so two replays of one directory are
    the same tuple.
    """
    root = Path(bench_dir) / CASES_DIRNAME
    if not root.is_dir():
        return ()
    found = [_case_artifacts(index) for index in sorted(root.rglob(JOB_NAME))]
    return tuple(sorted(found, key=lambda kept: (kept.case_id, kept.config_id, kept.rep)))


def _case_artifacts(index_file: Path) -> CaseArtifacts:
    """One ``job.json`` and the files it points at, as a :class:`CaseArtifacts`."""
    document = json.loads(index_file.read_text(encoding="utf-8"))
    directory = index_file.parent
    return CaseArtifacts(
        case_id=str(document["case_id"]),
        config_id=str(document["config_id"]),
        rep=int(document["rep"]),
        directory=directory,
        prompt=_read(directory / str(document["prompt_file"])) or "",
        attempts=tuple(_attempt_artifact(directory, attempt) for attempt in document["attempts"]),
    )


def _attempt_artifact(directory: Path, entry: Mapping[str, Any]) -> AttemptArtifact:
    """One indexed attempt, with the two files it may have written read back beside it."""
    output, error = entry["output_file"], entry["error_file"]
    return AttemptArtifact(
        index=int(entry["attempt"]),
        passed=bool(entry["passed"]),
        output=_read(directory / output) if output else None,
        error=_read(directory / error) if error else None,
    )


# ── one unit of work ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BenchJob:
    """One case, under one configuration, on one rep — and the two directories it owns."""

    case: Case
    config: ModelConfig
    rep: int
    directory: Path
    workdir: Path


@dataclass(frozen=True)
class WorkedJob:
    """One job as the executor finished it: the scored run, and the answers it collected.

    The replies travel *beside* the scored run rather than inside it because
    :class:`~noctis.eval.record.CaseRun` is the metrics module's input type and nothing else —
    widening it to carry answer text would put a benchmark's evidence in the arithmetic's way. Both
    halves are plain values, so a pooled worker returns one across the call queue intact.
    """

    run: CaseRun
    replies: tuple[str | None, ...] = ()


@dataclass(frozen=True)
class BenchRun:
    """What one finished bench run is: its identity, its directory, its plan and its record."""

    bench_id: str
    directory: Path
    plan: SpendPlan
    runs: tuple[CaseRun, ...]
    record: Mapping[str, Any]
    complete: bool


# ── the runner ────────────────────────────────────────────────────────────────────────────


def _payload_of(case: Case) -> Any:
    """The default site input: the case's payload, exactly as its file declared it.

    A site whose renderer takes a typed object (the coder's ``AuthoringJob``, the distiller's
    ``FindingsHistory``) needs a payload→input adapter, and building those is the CLI-wiring
    story's job. Naming the seam here is the honest way to leave that gap: an adapter is injected,
    never inferred, because guessing how a mapping becomes a site's input is how a benchmark starts
    measuring its own adapter.
    """
    return case.payload


def _utc_now() -> datetime:
    """The wall clock, in UTC — injectable, because a record's stamps are part of its contract."""
    return datetime.now(UTC)


@dataclass(frozen=True)
class BenchRunner:
    """Runs one corpus against one site under one harness spec, and writes one bench record.

    Everything that is not the runner's own arithmetic arrives injected: the model call
    (:data:`AttemptFn`), the site registry, the resolved identities the record quotes, the price
    table, the execution seam and the clock. That is what lets the whole suite drive a full bench
    end to end with no network — and it is the same posture the run store takes, for the same
    reason: the impure module should own the I/O and nothing else.
    """

    bench_root: Path
    attempt: AttemptFn
    registry: Mapping[str, AgentSite[Any, Any]] | None = None
    harness: HarnessSpec = HarnessSpec()
    executor: JobExecutor = SequentialExecutor()
    # How many times one rep may be asked before it is recorded as unsettled. One by default: a
    # retry policy is a measured variable, not a hidden default, and a bench that retried without
    # saying so would report a job-level pass rate its first-attempt rate cannot explain.
    max_attempts: int = 1
    # The resolved identities the record quotes verbatim. Each defaults to reading this checkout,
    # and each is injectable so a bench over a fabricated tree (or a stub site) is still honest
    # about what it measured.
    identity: SiteIdentity | None = None
    engine: EngineStamp | None = None
    corpus_version: str | None = None
    table: PriceTable | None = None
    taxonomy: FailureTaxonomy | None = None
    client_stack: Mapping[str, str | None] | None = None
    label: str | None = None
    # The per-attempt usage a preflight prices with. ``None`` — the honest default — yields no
    # dollar figure at all rather than a figure derived from a token count nobody supplied.
    tokens_per_attempt: Mapping[str, int] | None = None
    site_input: Callable[[Case], Any] = _payload_of
    clock: Callable[[], datetime] = _utc_now

    # ── the dry run ──────────────────────────────────────────────────────────────────────

    def preflight(
        self,
        corpus: Corpus,
        *,
        reps: int = 1,
        configs: Sequence[ModelConfig] = DEFAULT_CONFIGS,
        split: Split | None = None,
    ) -> SpendPlan:
        """State what this bench would run and what it would cost. Spends nothing, writes nothing.

        This is the dry run, and it is also the only place a plan is computed — :meth:`run` calls it
        rather than re-deriving one, so the numbers an operator agreed to are the numbers that
        execute. Every refusal a bench can raise before spending is raised here: an undeclared site,
        a knob the site does not accept, an id that would not stay inside its own directory.
        """
        declaration = self._declaration(corpus)
        # Refuse a bad override while it is still a dict — the knob module's whole purpose.
        self.harness.validate_knobs(declaration.knobs)
        _at_least_one("reps", reps)
        _at_least_one("max_attempts", self.max_attempts)
        _refuse_repeated_configs(configs)
        cases = _selected(corpus, split)
        for case in cases:
            _safe_name("case id", case.case_id)
        for config in configs:
            _safe_name("configuration id", config.config_id)
        return self._plan(corpus, cases, reps=reps, configs=tuple(configs), split=split)

    # ── the live run ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        corpus: Corpus,
        *,
        reps: int = 1,
        configs: Sequence[ModelConfig] = DEFAULT_CONFIGS,
        split: Split | None = None,
        acknowledged: SpendPlan | None = None,
    ) -> BenchRun:
        """Run the acknowledged plan, keep the evidence, and write one validated bench record.

        ``acknowledged`` is the plan :meth:`preflight` returned for exactly this shape. It is
        required — the minimal honest mechanism for "a live run refuses to start unless the
        preflight was run": there is no flag to forget, and a plan for another corpus, another rep
        count or another configuration set is refused by name rather than silently accepted.
        """
        plan = self.preflight(corpus, reps=reps, configs=configs, split=split)
        _acknowledge(plan, acknowledged)

        started = self.clock()
        bench_id = new_bench_id(started)
        directory = Path(self.bench_root) / bench_id
        directory.mkdir(parents=True)
        jobs = self._jobs(directory, _selected(corpus, split), tuple(configs), reps)
        worked = self.executor.run(self._work, jobs)
        finished = self.clock()

        runs = tuple(one.run for one in worked)
        # A partial bench must never pass for a whole one: every job worked, and every one of them
        # recorded an attempt. An errored attempt still ran; a job that produced none did not.
        complete = len(worked) == len(jobs) and all(run.result.ran for run in runs)
        record = self._record(
            corpus,
            bench_id=bench_id,
            runs=runs,
            answered=_answered(jobs, worked),
            complete=complete,
            configs=tuple(configs),
            split=split,
            started=started,
            finished=finished,
        )
        _write_json(directory / BENCH_RECORD_NAME, record)
        return BenchRun(
            bench_id=bench_id,
            directory=directory,
            plan=plan,
            runs=runs,
            record=record,
            complete=complete,
        )

    # ── one job ──────────────────────────────────────────────────────────────────────────

    def _work(self, job: BenchJob) -> WorkedJob:
        """Ask one case, keep every artifact it produced, and score nothing.

        A raising attempt callable becomes a failed attempt carrying its exception text, never a
        dead bench: one unreachable provider must not throw away the ninety-nine cases that already
        answered. The retry loop stops at the first pass, so ``attempts_to_pass`` means what the
        metrics module says it means.

        A renderer that raises is deliberately **not** caught the same way. A provider falling over
        is weather; a case whose prompt cannot be composed at all is a defect in the corpus or the
        site, and recording it as a failed attempt would publish a pass rate that silently counted
        asks nobody ever made.
        """
        job.workdir.mkdir(parents=True, exist_ok=True)
        prompt = self._prompt(job.case)
        (job.directory / PROMPT_NAME).write_text(prompt, encoding="utf-8")

        outcomes: list[AttemptOutcome] = []
        served: list[str | None] = []
        replies: list[str | None] = []
        index: list[Mapping[str, Any]] = []
        for number in range(1, self.max_attempts + 1):
            request = AttemptRequest(
                prompt=prompt,
                case=job.case,
                config=job.config,
                harness=self.harness,
                rep=job.rep,
                attempt=number,
                workdir=job.workdir,
            )
            answer = self._ask(request)
            outcomes.append(answer.outcome)
            served.append(answer.served_model)
            replies.append(answer.output)
            index.append(_keep_attempt(job.directory, number, answer))
            if answer.outcome.passed:
                break

        _write_json(
            job.directory / JOB_NAME,
            {
                "case_id": job.case.case_id,
                "config_id": job.config.config_id,
                "rep": job.rep,
                "prompt_file": PROMPT_NAME,
                "attempts": list(index),
            },
        )
        return WorkedJob(
            run=CaseRun(
                result=CaseResult(case_id=job.case.case_id, attempts=tuple(outcomes)),
                config_id=job.config.config_id,
                served_models=tuple(served),
            ),
            replies=tuple(replies),
        )

    def _ask(self, request: AttemptRequest) -> Attempt:
        """One model call, with a raising seam turned into the failed attempt it really is."""
        try:
            return self.attempt(request)
        except Exception as exc:  # one bad attempt ≠ a dead bench
            return Attempt(
                outcome=AttemptOutcome(passed=False, error=f"{type(exc).__name__}: {exc}")
            )

    def _prompt(self, case: Case) -> str:
        """The prompt this case is asked with — the site's own declared renderer, under the spec.

        The declaration is looked up per case rather than cached on the runner: a registry lookup is
        a dict read, and holding a resolved declaration as state is how a runner ends up rendering
        one bench's cases through another bench's site.
        """
        declaration = site(case.site_id, self.registry)
        return str(declaration.render(self.site_input(case), self.harness))

    # ── assembly ─────────────────────────────────────────────────────────────────────────

    def _jobs(
        self,
        directory: Path,
        cases: Sequence[Case],
        configs: Sequence[ModelConfig],
        reps: int,
    ) -> tuple[BenchJob, ...]:
        """Every job this bench will work, each with the two directories it owns, created empty."""
        jobs: list[BenchJob] = []
        for case in cases:
            for config in configs:
                for rep in range(1, reps + 1):
                    home = (
                        directory
                        / CASES_DIRNAME
                        / case.case_id
                        / config.config_id
                        / f"rep-{rep:02d}"
                    )
                    home.mkdir(parents=True)
                    jobs.append(
                        BenchJob(
                            case=case,
                            config=config,
                            rep=rep,
                            directory=home,
                            workdir=home / WORK_DIRNAME,
                        )
                    )
        return tuple(jobs)

    def _record(
        self,
        corpus: Corpus,
        *,
        bench_id: str,
        runs: Sequence[CaseRun],
        answered: Sequence[AnsweredCase],
        complete: bool,
        configs: Sequence[ModelConfig],
        split: Split | None,
        started: datetime,
        finished: datetime,
    ) -> Mapping[str, Any]:
        """Collect, build, validate — and refuse a document a reader could not trust."""
        artifacts = BenchArtifacts(
            bench_id=bench_id,
            site=self._identity(corpus),
            engine=self.engine if self.engine is not None else engine_stamp(),
            corpus=CorpusIdentity(
                version=self.corpus_version,
                hash=corpus.digest,
                case_count=len(corpus),
                split=WHOLE_CORPUS if split is None else split.value,
            ),
            harness_hash=harness_hash(self.harness),
            harness_dials=self._dials(corpus, answered),
            runs=tuple(runs),
            label=self.label,
            started_utc=_stamp(started),
            finished_utc=_stamp(finished),
            complete=complete,
            price_table=self.table,
            taxonomy=self.taxonomy,
            configs=tuple(configs),
            client_stack=self.client_stack,
        )
        record = build(artifacts)
        problems = validate(record)
        if problems:
            raise InvalidBenchRecord(
                "the collected artifacts do not build a readable bench record, so none was "
                f"written: {'; '.join(problems)}"
            )
        return record

    def _plan(
        self,
        corpus: Corpus,
        cases: Sequence[Case],
        *,
        reps: int,
        configs: tuple[ModelConfig, ...],
        split: Split | None,
    ) -> SpendPlan:
        """Count the jobs and price their ceiling — arithmetic over values, and nothing else."""
        table = self.table if self.table is not None else default_table()
        attempts = self.max_attempts
        jobs = len(cases) * reps * len(configs)
        ceiling = jobs * attempts
        per_config = len(cases) * reps * attempts
        usage = self.tokens_per_attempt
        return SpendPlan(
            site_id=corpus.site_id,
            corpus_digest=corpus.digest,
            split=WHOLE_CORPUS if split is None else split.value,
            cases=len(cases),
            reps=reps,
            configs=tuple(config.config_id for config in configs),
            max_attempts=attempts,
            jobs=jobs,
            attempts_ceiling=ceiling,
            tokens_ceiling=None if usage is None else sum(usage.values()) * ceiling,
            usd_ceiling_estimate=_ceiling_usd(table, configs, usage, per_config),
            pricing_table_version=table.version,
            unpriced_models=_unpriced(table, configs, usage),
        )

    def _dials(self, corpus: Corpus, answered: Sequence[AnsweredCase]) -> dict[str, object]:
        """The composed harness dials, plus every reading the site's declared scorers add.

        The site-agnostic scoring pass, and the whole of it: each declared scorer is handed every
        answered case and whatever mapping it returns is folded in verbatim. Nothing here knows what
        a reading contains — that is the site's business — so a second site's scorer lands in the
        record the day its declaration carries one, with no edit to this module.
        """
        dials = harness_dials(self.harness)
        for scorer in self._declaration(corpus).scorers:
            reading = scorer.read(answered)
            if reading is None:
                continue
            clash = sorted(set(reading) & set(dials))
            if clash:
                raise ConflictingReading(
                    f"a scorer of site {corpus.site_id!r} would overwrite the harness dial(s) "
                    f"{', '.join(repr(name) for name in clash)} — the composition says what was "
                    "asked and a reading says what the answers earned, so neither may rewrite the "
                    "other"
                )
            dials.update(reading)
        return dials

    def _declaration(self, corpus: Corpus) -> AgentSite[Any, Any]:
        """The site this corpus measures, resolved through the declarations registry."""
        return site(corpus.site_id, self.registry)

    def _identity(self, corpus: Corpus) -> SiteIdentity:
        """The site identity the record quotes — injected, or read off this checkout."""
        if self.identity is not None:
            return self.identity
        return site_identity(corpus.site_id, registry=self.registry)


# ── the arithmetic and the I/O helpers ────────────────────────────────────────────────────


def _answered(jobs: Sequence[BenchJob], worked: Sequence[WorkedJob]) -> tuple[AnsweredCase, ...]:
    """Every job paired with the answers it collected — the scoring pass's whole input.

    Paired positionally because that is the execution seam's contract: every
    :class:`JobExecutor` returns results *in job order*, so index ``i`` is job ``i``'s. A short
    tuple (an executor that dropped work) simply scores what really came back, which is the same
    posture ``complete`` takes over the same tuple.
    """
    return tuple(
        AnsweredCase(
            case=job.case,
            config_id=job.config.config_id,
            rep=job.rep,
            replies=one.replies,
        )
        for job, one in zip(jobs, worked, strict=False)
    )


def _acknowledge(plan: SpendPlan, acknowledged: SpendPlan | None) -> None:
    """Refuse a live bench whose spend nobody stated, or stated differently."""
    if acknowledged is None:
        raise SpendUnacknowledged(
            "a bench refuses to start on an unacknowledged spend — run preflight() and hand the "
            f"plan back as `acknowledged`. This one would be: {plan.summary()}"
        )
    if acknowledged != plan:
        raise SpendUnacknowledged(
            "the acknowledged plan differs from the one this run would execute, so nothing was "
            f"spent. Acknowledged: {acknowledged.summary()}. Would run: {plan.summary()}"
        )


def _at_least_one(knob: str, value: int) -> None:
    """Refuse a count below one rather than clamping it — a clamp is a small, silent lie.

    ``reps=0`` almost always means the caller computed it, and a bench that quietly ran once anyway
    would publish a record whose rep count nobody asked for.
    """
    if value < 1:
        raise ValueError(f"{knob} must be at least 1, got {value}")


def _refuse_repeated_configs(configs: Sequence[ModelConfig]) -> None:
    """Refuse two configurations answering to one id — the record could not attribute either."""
    ids = [config.config_id for config in configs]
    repeated = sorted({config_id for config_id in ids if ids.count(config_id) > 1})
    if not configs:
        raise ValueError("a bench runs under at least one model configuration, and states which")
    if repeated:
        raise ValueError(
            f"configuration id(s) {', '.join(repr(name) for name in repeated)} appear more than "
            "once — every result is attributed by config id, and two sharing one are unattributable"
        )


def _selected(corpus: Corpus, split: Split | None) -> tuple[Case, ...]:
    """The half of the corpus this bench measures — both halves when none is named."""
    if split is None:
        return corpus.cases
    return corpus.tuning if split is Split.TUNING else corpus.holdout


def _ceiling_usd(
    table: PriceTable,
    configs: Sequence[ModelConfig],
    usage: Mapping[str, int] | None,
    per_config_attempts: int,
) -> float | None:
    """The worst-case bill, or ``None`` the moment anything in it is unpriceable.

    All-or-nothing, exactly as :meth:`~noctis.research.pricing.PriceTable.total_usd` is: a ceiling
    that quietly dropped the configuration it could not price would read as complete while
    understating the spend an operator is about to agree to.
    """
    if usage is None:
        return None
    total = 0.0
    for config in configs:
        one = table.estimate_usd(config.requested_model or "", usage)
        if one is None:
            return None
        total += one * per_config_attempts
    return round(total, 6)


def _unpriced(
    table: PriceTable, configs: Sequence[ModelConfig], usage: Mapping[str, int] | None
) -> tuple[str, ...]:
    """Every configuration's model the table could not price, named so the ``n/a`` is explicable."""
    if usage is None:
        return ()
    return tuple(
        sorted(
            {
                config.requested_model or UNSTATED_MODEL
                for config in configs
                if table.estimate_usd(config.requested_model or "", usage) is None
            }
        )
    )


def _keep_attempt(directory: Path, number: int, answer: Attempt) -> Mapping[str, Any]:
    """Write one attempt's output and error beside its case, and index what was written."""
    output_file = f"attempt-{number:02d}.output.txt" if answer.output is not None else None
    error_file = f"attempt-{number:02d}.error.txt" if answer.outcome.error is not None else None
    if output_file is not None and answer.output is not None:
        (directory / output_file).write_text(answer.output, encoding="utf-8")
    if error_file is not None and answer.outcome.error is not None:
        (directory / error_file).write_text(answer.outcome.error, encoding="utf-8")
    return {
        "attempt": number,
        "passed": answer.outcome.passed,
        "model": answer.outcome.model,
        "served_model": answer.served_model,
        "latency_s": answer.outcome.seconds,
        "usage": dict(answer.outcome.usage()),
        "output_file": output_file,
        "error_file": error_file,
    }


def _safe_name(kind: str, name: str) -> str:
    """Refuse an id that would not stay inside its own directory. Isolation, before it is tested.

    A case id names a directory, so an id carrying a separator (or spelled ``..``) would place one
    case's artifacts on top of another's — or outside the bench area entirely. Refused by name
    rather than sanitized: silently rewriting an id would make the record disagree with the corpus.
    """
    if not name.strip() or name in {".", ".."} or set(name) & set("/\\") or "\x00" in name:
        raise UnsafeArtifactName(
            f"{kind} {name!r} would not stay inside its own directory — a bench keeps one "
            "artifact area per case, named by the id itself"
        )
    return name


def _stamp(moment: datetime) -> str:
    """One observed instant as the record spells it: UTC ISO-8601 with a ``Z``."""
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
    return aware.astimezone(UTC).strftime(_STAMP_FORMAT)


def _read(path: Path) -> str | None:
    """One retained artifact, or ``None`` when it was never written."""
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _write_json(target: Path, document: Mapping[str, Any]) -> None:
    """One atomic JSON write — the run store's discipline: a temp file, then ``os.replace``.

    Deliberately re-stated here rather than imported: the :mod:`noctis.reporting.run_tree` package
    is the only code allowed to touch the runs tree, and a bench runner reaching into it — even for
    the ten lines of :mod:`noctis.reporting.run_tree.record` — would blur exactly the boundary this
    module exists to keep.
    """
    tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(document, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _noctis_version() -> str:
    """The package literal — informational beside the engine version, never a comparison key."""
    from importlib import metadata

    try:
        return metadata.version("noctis")
    except Exception:  # not pip-installed (editable/source tree)
        from noctis import __version__

        return __version__
