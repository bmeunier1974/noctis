"""The retrospective DECIDE miner — a corpus harvested from real runs, and a bench costing nothing.

Every DECIDE verdict this system has ever spent is already on disk: the strategy's journal holds the
evidence the briefing was built from and the verdict that answered it, the session ledger holds the
deferrals beside it, and the run record holds the configuration both were produced under. This
module turns that history into the two artifacts a benchmark needs — **a corpus** (one reviewable
YAML case per decided candidate, in the layout
:class:`~noctis.eval.case_provider.YamlCaseProvider` already reads) and **a baseline record**
(``bench.json`` over those cases' *recorded* answers) — and it does both without asking a model
anything at all.

**It is the I/O half of #207, and only that.** Every judgment about what a case says lives in the
pure builder :func:`~noctis.eval.decide_case.build_decide_case`: the tiered label, the difficulty
axes, the ``n/a`` an unreplayable arbitration earns, the ``None`` an undecided candidate earns. This
module reads files, hands the records over, counts what came back, and writes. That split is why the
mining rules are testable with a pencil and why nothing here can quietly invent a label.

**Deterministic and idempotent, because a corpus is a population.** Runs are walked in id order and
candidates in name order; a case is serialized as canonical sorted-key YAML; a file already on disk
with identical bytes is **left untouched** (its mtime does not move) and reported ``unchanged``. So
re-mining a workspace never duplicates a case, never rewrites one for cosmetic reasons, and never
makes yesterday's benchmark number depend on when it was last mined.

**Nothing is guessed, and every absence is reported.** Three things make a run unminable and each is
named in :class:`SkippedRun` rather than papered over: a directory whose name is not a run id the
engine minted (the reserved ``legacy`` tree is one — a case's provenance *is* a minted run id), a
run with no readable record, and a run whose record froze no configuration. That last one matters
most: the exhaustion floor and the promotion rules are part of the *ask*, so mining a run without
them would freeze a prompt that run never saw.

**The retrospective record: what it claims, and what it refuses to claim.**
:func:`write_retrospective_bench` scores the corpus's recorded verdicts through
:func:`~noctis.eval.decide_scorer.score_decide_batch` and publishes them through the same pure
builder every bench record goes through (:func:`noctis.eval.record.build`), with three deliberate
decisions a reader should not have to reverse-engineer:

1. **``results`` holds exactly the labelled approvals.** Approval-side agreement is defined over the
   candidates the gates actually judged, so those are the rows whose single "attempt" carries a
   pass/fail — ``passed`` is the gates' own disposition. A rejection has no gate counterfactual and
   a deferral never reached the gates; forcing either through a pass rate would publish a number
   about a population nobody measured. They stay in the corpus (``corpus.case_count`` states the
   whole of it, beside ``bench.cases``, so the gap is visible) and are reported in the dials.
2. **The whole DECIDE reading rides in ``harness.dials``**, the one subtree the record quotes
   verbatim: the co-primary :class:`~noctis.eval.decide_scorer.ApprovalPair`, the deferral figures,
   one row per mined case with its axes, and the per-axis strata scored the same way. The scorer is
   the source of truth for every one of those numbers; the headline ``job_pass_rate`` the record
   computes from ``results`` equals its ``agreement`` by construction.
3. **The harness hash is ``null``, so the key is incomplete and the record is never subtractable.**
   The compositions that produced these answers are as many as the runs they were mined from, and
   none of them is this checkout's. Claiming one would let a retrospective baseline be differenced
   against a live bench of today's prompts, which is the single most inviting wrong subtraction on
   offer here. :func:`~noctis.eval.record.side_by_side` therefore refuses it structurally.

**Zero spend is structural, not a promise.** There is no attempt callable on this path, no client
seam is imported, and every recorded attempt states a measured zero token usage — the answers were
paid for months ago, and re-reading them costs nothing.

**Boundary.** Writes land under the cases root and the bench area, and nowhere else. The runs tree
is read-only input (its readers — :mod:`noctis.reporting.run_tree.record`, the journal, the
ledger — are the engine's own), no run lock is taken, and the champions registry is never
opened.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from noctis.champions.promotion import PromotionRules
from noctis.eval.case import Case, parse_case
from noctis.eval.case_provider import CASE_SUFFIX, YamlCaseProvider
from noctis.eval.corpus import Corpus
from noctis.eval.decide_case import (
    LABEL_PROMOTED,
    LABEL_REFUSED,
    SITE_ID,
    decide_case_document,
    decide_case_id,
    read_outcome,
)
from noctis.eval.decide_scorer import (
    DecideMetrics,
    DecideOutcome,
    GateLabel,
    score_decide_batch,
)
from noctis.eval.decide_site import (
    ANSWERS_RECORDED,
    DECIDE_DIALS_KEY,
    case_row,
    scored_block,
    strata_block,
)
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
from noctis.eval.runner import (
    BENCH_RECORD_NAME,
    WHOLE_CORPUS,
    InvalidBenchRecord,
    engine_stamp,
    new_bench_id,
)
from noctis.eval.site import AgentSite
from noctis.observability.debug.runid import RUN_ID_RE
from noctis.reporting.run_tree.record import read_record
from noctis.research.driver import DECIDE
from noctis.research.journal import ExperimentJournal
from noctis.research.ledger import SESSIONS_DIRNAME, SessionLedger
from noctis.research.pricing import PriceTable

__all__ = [
    "MiningReport",
    "RetrospectiveBench",
    "SkippedRun",
    "decide_outcomes",
    "load_decide_corpus",
    "mine_decide_corpus",
    "retrospective_dials",
    "write_retrospective_bench",
]

# ── the skip reasons, spelled once so a report and its test read the same words ──────────────

UNMINTED_RUN_ID = (
    "the directory name is not a run id the engine minted, and a mined case's provenance is one"
)
NO_RECORD = "the run has no readable run.json, so nothing states what it was configured to do"
NO_FROZEN_CONFIG = (
    "the run's record froze no configuration, and the exhaustion floor and promotion rules it "
    "decided under are part of the ask rather than defaults to assume"
)


@dataclass(frozen=True)
class SkippedRun:
    """One run directory the miner did not read, and the honest reason it did not."""

    run_id: str
    reason: str


@dataclass(frozen=True)
class MiningReport:
    """What one mining pass did: the corpus it produced, and everything it left behind.

    ``cases`` is every case this workspace yields — written this pass or already on disk — in
    case-id order, so the report *is* the corpus. ``mined`` names the files this pass wrote,
    ``unchanged`` the ones it found byte-identical and did not touch, and ``excluded`` the episodes
    that carry no reconstructable outcome (the ``n/a`` tier: an undecided candidate is counted,
    never labelled). ``requested_models`` is each mined run's research model, read off its own
    record, for the retrospective bench's provenance tier.
    """

    cases: tuple[Case, ...] = ()
    mined: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    skipped_runs: tuple[SkippedRun, ...] = ()
    requested_models: Mapping[str, str | None] = field(default_factory=dict)

    def summary(self) -> str:
        """The one line an operator reads after a mining pass."""
        return (
            f"{len(self.cases)} case(s) in the {SITE_ID} corpus: {len(self.mined)} mined, "
            f"{len(self.unchanged)} unchanged, {len(self.excluded)} excluded (n/a), "
            f"{len(self.skipped_runs)} run(s) skipped"
        )


@dataclass(frozen=True)
class RetrospectiveBench:
    """One published retrospective baseline: where it landed, what it says, and how it scored."""

    bench_id: str
    directory: Path
    record: Mapping[str, Any]
    metrics: DecideMetrics


# ── mining ───────────────────────────────────────────────────────────────────────────────────


def mine_decide_corpus(runs_root: Path | str, *, cases_root: Path | str) -> MiningReport:
    """Harvest every decided candidate under ``runs_root`` into ``<cases_root>/decide/*.yaml``.

    ``runs_root`` is the workspace's runs tree (``<workspace>/runs``), read and never written;
    ``cases_root`` is the corpus root the provider loads from, and the only place this writes.
    Runs are walked in id order and candidates in name order, so two mining passes over one
    workspace produce the same files in the same bytes — and a file already holding those bytes is
    left exactly as it is.
    """
    runs = Path(runs_root)
    directory = Path(cases_root) / SITE_ID
    cases: list[Case] = []
    mined: list[str] = []
    unchanged: list[str] = []
    excluded: list[str] = []
    skipped: list[SkippedRun] = []
    models: dict[str, str | None] = {}

    for run_dir in sorted(p for p in runs.iterdir() if p.is_dir()) if runs.is_dir() else []:
        inputs = _run_inputs(run_dir)
        if isinstance(inputs, SkippedRun):
            skipped.append(inputs)
            continue
        models[run_dir.name] = inputs.research_model
        for episode in _mine_run(run_dir, inputs.settings, directory):
            if episode.case is None:
                excluded.append(episode.case_id)
                continue
            cases.append(episode.case)
            (mined if episode.written else unchanged).append(episode.case_id)

    return MiningReport(
        cases=tuple(sorted(cases, key=lambda case: case.case_id)),
        mined=tuple(mined),
        unchanged=tuple(unchanged),
        excluded=tuple(excluded),
        skipped_runs=tuple(skipped),
        requested_models=dict(models),
    )


def load_decide_corpus(
    cases_root: Path | str, *, registry: Mapping[str, AgentSite[Any, Any]] | None = None
) -> Corpus:
    """The mined corpus, loaded back through the eval core's own provider and split once.

    Deliberately thin: mining writes the interchange format, and reading it is the provider's job,
    so a mined corpus is admitted on exactly the terms a hand-written one is — validation, site
    cross-check, the deterministic split — and there is no second reader to drift from the first.
    """
    cases = YamlCaseProvider(cases_root=Path(cases_root), registry=registry).load(SITE_ID)
    return Corpus(site_id=SITE_ID, cases=cases)


@dataclass(frozen=True)
class _Episode:
    """One candidate of one run, after mining: the case it yielded, or the absence of one."""

    case_id: str
    case: Case | None = None
    written: bool = False


def _mine_run(run_dir: Path, settings: SimpleNamespace, directory: Path) -> list[_Episode]:
    """Every candidate one run journaled, in name order, mined into the corpus directory."""
    run_id = run_dir.name
    state = _state_dir(run_dir)
    journal = ExperimentJournal(state)
    rules = PromotionRules.from_settings(settings)
    minimum = int(settings.research.min_trials)
    episodes: list[_Episode] = []
    for strategy in journal.strategies():
        case_id = decide_case_id(strategy, run_id)
        document = decide_case_document(
            strategy,
            run_id=run_id,
            journal_records=journal.records(strategy),
            ledger_records=_deciding_session(state, strategy),
            min_trials=minimum,
            rules=rules,
            # The board of that moment is not recoverable from the tree, and #207's replay is
            # trusted only when it agrees with the recorded disposition — so an empty board yields
            # gate axes where the arbitration reproduces, and ``n/a`` where it does not.
            champions=(),
        )
        if document is None:
            episodes.append(_Episode(case_id=case_id))
            continue
        path = directory / f"{case_id}{CASE_SUFFIX}"
        case = parse_case(document, case_id=case_id, source=str(path))
        episodes.append(_Episode(case_id=case_id, case=case, written=_write_case(path, document)))
    return episodes


def _write_case(path: Path, document: Mapping[str, Any]) -> bool:
    """Write one case file, or leave an identical one alone. ``True`` when it wrote.

    Canonical YAML — sorted keys, block style, unicode as itself — so the bytes depend on the case
    and on nothing else. The read-before-write is what makes re-mining a no-op rather than a
    timestamp churn: a corpus file that did not change should not look like it did in a diff.
    """
    text = yaml.safe_dump(_plain(document), sort_keys=True, allow_unicode=True)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def _deciding_session(state: Path, strategy: str) -> list[dict[str, Any]]:
    """The ledger of the session that decided ``strategy`` — the tail its briefing folded in.

    A DECIDE briefing is built from the session running it, so the case freezes that session's
    ledger and no other. The scope is the ledger's own: the DECIDE ``stage`` line naming the
    strategy. A candidate no session's DECIDE stage names contributes no tail at all rather than a
    foreign session's narrative; the latest matching session wins, because a re-decided candidate's
    case is the one it was last asked about.
    """
    sessions = sorted((state / SESSIONS_DIRNAME).glob("*.jsonl"))
    for path in reversed(sessions):
        records = SessionLedger.from_path(path).records()
        if any(
            record.get("event") == "stage"
            and record.get("stage") == DECIDE
            and record.get("strategy") == strategy
            for record in records
        ):
            return records
    return []


@dataclass(frozen=True)
class _RunInputs:
    """What one run must state before it can be mined faithfully — read from its record, once."""

    settings: SimpleNamespace
    research_model: str | None


def _run_inputs(run_dir: Path) -> _RunInputs | SkippedRun:
    """This run's frozen ask, or the honest reason it cannot be mined.

    The three refusals are stated together because they are one question — *does this directory say
    what it was configured to do?* — and every one of them is a skip rather than a default, since a
    default would freeze a prompt no session was ever shown.
    """
    if not RUN_ID_RE.match(run_dir.name):
        return SkippedRun(run_id=run_dir.name, reason=UNMINTED_RUN_ID)
    record, _ = read_record(run_dir)
    if record is None:
        return SkippedRun(run_id=run_dir.name, reason=NO_RECORD)
    settings = _frozen_settings(record)
    if settings is None:
        return SkippedRun(run_id=run_dir.name, reason=NO_FROZEN_CONFIG)
    return _RunInputs(settings=settings, research_model=_research_model(record))


def _frozen_settings(record: Mapping[str, Any]) -> SimpleNamespace | None:
    """The run's frozen configuration as the duck :class:`PromotionRules` reads, or ``None``.

    The record's ``inputs.settings.resolved`` block *is* a settings dump, so the values are read
    from it directly rather than by rehydrating a whole ``Settings`` (which would need a live
    settings object, the environment and the path knobs this module has no business touching). Only
    what a DECIDE ask is shaped by is taken: the exhaustion floor, and the promotion rules — the
    thresholds plus the board size they arbitrate against. A record missing any of them states no
    configuration at all as far as this module is concerned — see :data:`NO_FROZEN_CONFIG`.
    """
    resolved = _section(_section(_section(record, "inputs"), "settings"), "resolved")
    promotion = _section(resolved, "promotion")
    research = _section(resolved, "research")
    if not promotion or "min_trials" not in research or "champion_count" not in resolved:
        return None
    settings = SimpleNamespace(
        champion_count=resolved["champion_count"],
        promotion=SimpleNamespace(**promotion),
        research=SimpleNamespace(min_trials=research["min_trials"]),
    )
    try:
        PromotionRules.from_settings(settings)
    except (AttributeError, TypeError):  # a record whose promotion block predates a threshold
        return None
    return settings


def _research_model(record: Mapping[str, Any]) -> str | None:
    """The model this run researched with, as its record resolved it — ``None`` when it named none.

    The record's ``inputs.models`` block is the run's own resolved answer to "which model drove a
    session", so it is read rather than re-derived from the settings dump beside it.
    """
    model = _section(_section(record, "inputs"), "models").get("research")
    return str(model) if model else None


def _section(node: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """One nested mapping of a record, or an empty one — the tolerant read, stated once."""
    value = node.get(key) if isinstance(node, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _state_dir(run_dir: Path) -> Path:
    """The run's state tree, through the one derivation that owns the run-scoped layout."""
    from noctis.config.settings import run_scoped_paths

    return Path(run_scoped_paths(run_dir)["state_dir"])


def _plain(value: Any) -> Any:
    """A case document as plain YAML-safe data: read-only maps and tuples undone."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


# ── the zero-spend retrospective record ──────────────────────────────────────────────────────


def decide_outcomes(cases: Sequence[Case]) -> tuple[DecideOutcome, ...]:
    """The corpus's *recorded* answers, as the scorer's input type — history, not a re-run.

    A case carrying no recorded outcome contributes none: it was never decided, so it belongs to no
    denominator here. The mapping is one-to-one and lossless — the eventual verdict, the gates'
    label (``UNLABELED`` when they never gave one), whether a deferral happened and whether it moved
    the answer.
    """
    scored: list[DecideOutcome] = []
    for case in cases:
        recorded = read_outcome(case)
        if recorded is None:
            continue
        scored.append(
            DecideOutcome(
                case_id=case.case_id,
                verdict=recorded.eventual_verdict,
                label=_gate_label(recorded.label),
                revised=recorded.revises > 0,
                revise_flipped=recorded.revise_flip,
            )
        )
    return tuple(scored)


def write_retrospective_bench(
    corpus: Corpus,
    *,
    bench_root: Path | str,
    requested_models: Mapping[str, str | None] | None = None,
    label: str | None = None,
    identity: SiteIdentity | None = None,
    engine: EngineStamp | None = None,
    corpus_version: str | None = None,
    table: PriceTable | None = None,
    now: datetime | None = None,
) -> RetrospectiveBench:
    """Score a mined corpus's recorded verdicts and publish one validated ``bench.json``.

    Spends nothing: there is no attempt callable, no prompt is rendered and no model is asked. The
    record is built by :func:`noctis.eval.record.build` and checked by
    :func:`~noctis.eval.record.validate` before anything is written, exactly as the live runner does
    — a document a reader could not trust is refused rather than published.

    ``requested_models`` maps a run id to the model that run researched with (a
    :class:`MiningReport` carries it); an unknown run is configured with ``None`` rather than with a
    plausible alias.
    """
    outcomes = decide_outcomes(corpus.cases)
    metrics = score_decide_batch(outcomes)
    started = now if now is not None else datetime.now(UTC)
    bench_id = new_bench_id(started)

    artifacts = BenchArtifacts(
        bench_id=bench_id,
        site=identity if identity is not None else site_identity(SITE_ID),
        engine=engine if engine is not None else engine_stamp(),
        corpus=CorpusIdentity(
            version=corpus_version,
            hash=corpus.digest,
            case_count=len(corpus),
            split=WHOLE_CORPUS,
        ),
        # Deliberately null — see the module docstring. The compositions behind these answers are
        # as many as the runs they were mined from, so this record carries an incomplete key and is
        # structurally incomparable to a live bench.
        harness_hash=None,
        harness_dials=retrospective_dials(corpus.cases, outcomes, metrics),
        runs=_graded_runs(corpus.cases, outcomes),
        label=label,
        started_utc=_stamp(started),
        finished_utc=_stamp(started),
        # Nothing was interrupted and nothing is outstanding: every answer was recorded before this
        # bench began.
        complete=True,
        price_table=table,
        configs=_configs(corpus.cases, requested_models or {}),
    )
    record = build(artifacts)
    problems = validate(record)
    if problems:  # pragma: no cover - a builder/validator disagreement is a defect, not a state
        raise InvalidBenchRecord(
            "the mined corpus does not build a readable bench record, so none was written: "
            f"{'; '.join(problems)}"
        )

    directory = Path(bench_root) / bench_id
    directory.mkdir(parents=True)
    _write_json(directory / BENCH_RECORD_NAME, record)
    return RetrospectiveBench(
        bench_id=bench_id, directory=directory, record=record, metrics=metrics
    )


def retrospective_dials(
    cases: Sequence[Case], outcomes: Sequence[DecideOutcome], metrics: DecideMetrics
) -> dict[str, Any]:
    """The whole DECIDE reading, as the verbatim block a bench record quotes under ``harness``.

    Everything a reader needs to recompute the headline and to stratify it: the co-primary pair, the
    deferral figures, one row per mined case (its verdict, its label, its difficulty axes) and the
    same scored block per axis level. The scorer's arithmetic is the only arithmetic here.
    """
    return {
        # The three facts that distinguish this record from a live bench, stated up front.
        "retrospective": True,
        "answers": ANSWERS_RECORDED,
        "attempt_calls": 0,
        DECIDE_DIALS_KEY: {
            **scored_block(metrics),
            "cases": [_case_row(case) for case in sorted(cases, key=lambda one: one.case_id)],
            "strata": strata_block(cases, outcomes),
        },
    }


def _case_row(case: Case) -> dict[str, Any]:
    """One mined case as the record lists it: where it came from, and what history said.

    The row itself is shaped by :func:`~noctis.eval.decide_site.case_row`, which the live scoring
    pass shapes its rows with too — the two readings share their keys structurally rather than by
    a convention that would drift the first time one of them grew a field.
    """
    recorded = read_outcome(case)
    return case_row(
        case,
        verdict=None if recorded is None else recorded.eventual_verdict,
        label=None if recorded is None else recorded.label,
        revises=0 if recorded is None else recorded.revises,
        revise_flip=False if recorded is None else recorded.revise_flip,
    )


def _graded_runs(cases: Sequence[Case], outcomes: Sequence[DecideOutcome]) -> tuple[CaseRun, ...]:
    """One row per **labelled approval** — the population approval-side agreement is defined over.

    Its single attempt is the recorded historical verdict, and ``passed`` is the gates' disposition
    of it, so the record's ``job_pass_rate`` *is* the scorer's agreement. Everything else the corpus
    holds — rejections, deferrals, approvals the gates never judged — has no approval-side pass or
    fail and is reported in the dials instead of being forced through a rate it cannot mean.

    The usage split is four measured zeros: this bench asked nobody anything, so it spent nothing.
    The duration stays unrecorded, because how long a *recorded* answer took is not something this
    bench measured.
    """
    origin = {case.case_id: case.provenance.mined_from for case in cases}
    return tuple(
        CaseRun(
            result=CaseResult(
                case_id=outcome.case_id,
                attempts=(_recorded_attempt(passed=outcome.label is GateLabel.PROMOTED),),
            ),
            config_id=origin.get(outcome.case_id),
        )
        for outcome in outcomes
        if outcome.approved and outcome.labeled
    )


def _recorded_attempt(*, passed: bool) -> AttemptOutcome:
    """One historical verdict, as the attempt a bench record accounts for.

    Every usage field is a **measured zero**, not an unknown: this bench made no model call, so it
    really did spend nothing, and the record's cost block should say so rather than report an
    absence. ``seconds`` stays unrecorded for the opposite reason — how long a recorded answer took
    is a fact about the session that spent it, not something re-reading it measured. ``model`` is
    ``None`` because nobody was asked.
    """
    return AttemptOutcome(
        passed=passed,
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )


def _configs(
    cases: Sequence[Case], requested_models: Mapping[str, str | None]
) -> tuple[ModelConfig, ...]:
    """One configuration per run the corpus was mined from — the historical production session.

    That *is* the configuration here: these verdicts were produced by a run, under the model that
    run researched with. The alias is quoted from the run's own record where it is recoverable and
    left ``None`` where it is not; no served model is ever recorded, because nothing on this path
    asked a provider anything.
    """
    runs = sorted({case.provenance.mined_from for case in cases if case.provenance.mined_from})
    return tuple(
        ModelConfig(config_id=run_id, requested_model=requested_models.get(run_id))
        for run_id in runs
    )


def _gate_label(label: str | None) -> GateLabel:
    """One recorded disposition as the scorer's label — an absent one is honestly unlabelled."""
    if label == LABEL_PROMOTED:
        return GateLabel.PROMOTED
    if label == LABEL_REFUSED:
        return GateLabel.REFUSED
    return GateLabel.UNLABELED


def _stamp(moment: datetime) -> str:
    """One instant as every Noctis record spells it: UTC ISO-8601 with a ``Z``."""
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
    return aware.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(target: Path, document: Mapping[str, Any]) -> None:
    """One atomic JSON write — a temp file, then ``os.replace``, the run store's discipline.

    Restated rather than reached for, exactly as :mod:`noctis.eval.runner` restates it: the modules
    that own a tree own the writing of it, and borrowing another module's private writer is how a
    boundary starts to blur.
    """
    tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(document, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
