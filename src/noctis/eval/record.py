"""``bench.json`` — the bench record: a pure builder, a pure validator, one key, one refusal.

A benchmark number is worthless without the conditions it was produced under, and *worse* than
worthless when two numbers produced under different conditions are subtracted. This module is the
run record's discipline applied to a bench run (``reporting/run_record.py`` beside
``reporting/schema.py``, deliberately mirrored rather than reinvented — and, for the conventions
that are about a *document* rather than about a run, imported from it outright: the generic
walkers behind :func:`validate` are the run record's own, exported by ``reporting/schema.py``):

* the **runner collects**, the **builder builds**, the **validator checks** — three jobs, and only
  the first of them is allowed to touch a disk or a clock. :func:`build` is a pure function from a
  frozen :class:`BenchArtifacts` to one JSON document, which is what makes the golden-record
  snapshot below cheap and the equivalence arguments provable rather than integration-tested;
* every aggregate is **derived at build time** by handing the per-case artifacts to
  :mod:`noctis.eval.metrics` and :mod:`noctis.eval.taxonomy`, never incremented as the bench runs.
  Two consequences worth stating: rebuilding the same artifacts reproduces the record byte for
  byte, and a reader holding the record can recompute every published figure from the ``results``
  section beside it;
* a known-absent value is an explicit ``null`` — never an omitted key, never a zero. A *measured*
  zero (nothing passed) is a finding; a *missing* input (nobody classified the failures, the price
  table could not price a model) is ``null``, and it propagates: a figure whose input is unknown
  stays unknown all the way into the comparison;
* :data:`SCHEMA_VERSION` is **additive-only**, and additive means *declared*. New keys may arrive
  at any time and an existing key never changes meaning; the record's sections are a floor rather
  than a ceiling, so a later story's whole new section is readable here. But a section's declared
  keys are the whole of what it may carry, and :func:`validate` — the run record's two-way keys
  walker, story #350 — names a key nobody declared rather than publishing an emitter typo as a
  field no reader indexes.

**The key is the point of the record.** :func:`comparable_key` composes one greppable label out of
the seven things that have to have been equal before two bench numbers may be subtracted — the
engine that scored, the site and generation that answered, the prompt text it was asked with, the
corpus it was asked over, the harness composition, and the price table the dollars came from. It
follows the engine-identity precedent (``observability/engine_id.py``): digests beside declared
versions, folded into one label a human can grep an issue for. Model identity is deliberately
**not** in it — see the provenance tier below.

**The refusal, not the comparison.** :func:`side_by_side` is this story's whole comparison surface:
equal keys yield a value carrying both figure sets and their deltas; differing (or incompletely
known) keys yield a value carrying a loud banner and, *structurally*, no delta field at all. The
paired-statistics compare verb belongs to its own epic; what belongs here is the guarantee that no
consumer can subtract two records that were never comparable, because the type it is handed has
nothing to subtract.

**The provenance tier: recorded, never keyed.** The requested provider/model per config, the
*served* model id per attempt, and the client-stack versions all ride in the record and none of
them enters the key. That asymmetry is deliberate. Keying on model identity would fragment every
comparison bucket the day a provider moved a dated snapshot under an alias — but a bench whose
alias silently began resolving to a different snapshot mid-run is exactly the thing an operator
must be able to *see*, so each config publishes the distinct served ids observed under it and the
``alias_drift`` flag derived from them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, TypeAlias

from noctis.eval.identity import SiteIdentity
from noctis.eval.metrics import AttemptOutcome, BenchMetrics, CaseResult, compute_metrics
from noctis.eval.taxonomy import UNCLASSIFIED, FailureTaxonomy, UnknownVocabulary
from noctis.reporting.schema import (
    check_estimate_labels,
    check_keys,
    check_stamps,
    check_units,
)
from noctis.research.pricing import USAGE_FIELDS, PriceTable, default_table

__all__ = [
    "HEADLINE_FIGURES",
    "KEY_COMPONENTS",
    "KEY_SEPARATOR",
    "KIND",
    "REQUIRED_SECTIONS",
    "SCHEMA_VERSION",
    "UNKNOWN",
    "BenchArtifacts",
    "CaseRun",
    "ComparableKey",
    "ComparablePair",
    "Comparison",
    "CorpusIdentity",
    "EngineStamp",
    "IncomparablePair",
    "ModelConfig",
    "build",
    "comparable",
    "comparable_key",
    "engine_digest",
    "key_of",
    "side_by_side",
    "validate",
]

# The bench record's contract version. Additive-only: bump it only for a breaking change, exactly
# as the run record's does.
SCHEMA_VERSION = 1

# The record's self-declared type, so a consumer can tell a bench record from a run record at a
# glance — the two are deliberately similar in shape and must never be confused for one another.
KIND = "noctis.bench"

# The one character the key label joins its components with, and the token an unknown component
# renders as. ``null`` rather than an empty string, because "we could not identify this" must read
# differently from "this was empty".
KEY_SEPARATOR = "|"
UNKNOWN = "null"

# A truncated SHA-256, the engine fingerprint's own convention: long enough that a collision is not
# a realistic concern, short enough to paste into an issue.
_DIGEST_CHARS = 16

# The top-level sections every bench record carries. Additive-only means this tuple may grow, never
# shrink. ``failures`` is here as an explicit ``null``-able section: a bench nobody registered a
# failure vocabulary for reports *no* breakdown, rather than a breakdown whose every count is zero.
REQUIRED_SECTIONS = (
    "bench",
    "site",
    "corpus",
    "harness",
    "engine",
    "comparable_key",
    "metrics",
    "failures",
    "results",
    "provenance",
)

# The figures a side-by-side view puts next to each other. Deliberately few: this is the headline
# an operator reads, not the whole metrics block, which both records carry in full anyway.
HEADLINE_FIGURES = (
    "first_attempt_pass_rate",
    "job_pass_rate",
    "mean_attempts_to_pass",
    "usd_per_pass_estimate",
    "p50_s",
)

# ── the artifacts: dumb, frozen data the runner collects ───────────────────────────────────


@dataclass(frozen=True)
class EngineStamp:
    """Which engine scored this bench: the declared version and the per-component digests.

    Both, for the reason ``observability/engine_id.py`` states — a declared version can be
    forgotten in review, a digest cannot; a digest cannot say whether a difference *matters*, a
    declared version can. The whole component map is recorded, so a reader can see which surface
    moved, and :func:`engine_digest` folds it into the one value the key carries.
    """

    engine_version: int
    fingerprint: Mapping[str, str | None] = field(default_factory=dict)
    noctis_version: str | None = None


@dataclass(frozen=True)
class CorpusIdentity:
    """Which corpus was asked, as the resolved strings the runner already holds.

    Taken as plain values rather than computed here: the corpus module owns how a corpus hashes
    itself, and a builder that recomputed it would be a second answer to that question.
    ``case_count`` is the corpus's declared size, which is deliberately **not** the number of cases
    this bench ran — the two differing is exactly the coverage gap a reader wants to see.
    """

    version: str | None = None
    hash: str | None = None
    case_count: int | None = None
    split: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    """One requested provider/model configuration a bench ran its cases under.

    ``requested_model`` is the alias asked for; what actually answered is per attempt (see
    :attr:`CaseRun.served_models`). Provenance tier: recorded, never keyed.
    """

    config_id: str
    provider: str | None = None
    requested_model: str | None = None


@dataclass(frozen=True)
class CaseRun:
    """One rep — the scored result the metrics read, plus the provenance the key never touches.

    ``result`` is the metrics module's own input type, untouched, so the aggregates in the record
    are that module's arithmetic rather than a second implementation of it. ``served_models`` is
    positionally aligned with ``result.attempts``; a shorter tuple leaves the remaining attempts'
    served id unknown, and an unknown served id is never back-filled from the alias that was asked
    for — that inference is precisely the drift this tier exists to make visible.
    """

    result: CaseResult
    config_id: str | None = None
    served_models: tuple[str | None, ...] = ()

    def served(self, index: int) -> str | None:
        """The served model id of attempt ``index``, or ``None`` when none was recorded."""
        return self.served_models[index] if index < len(self.served_models) else None


@dataclass(frozen=True)
class BenchArtifacts:
    """Everything one bench run knows about itself — the builder's only input.

    Every value here is already **resolved**: the site identity has been read off the tree, the
    corpus and harness hashes computed, the timestamps observed by the runner. That is what keeps
    :func:`build` free of a clock and a filesystem, and it is why the runner is the only side of
    this seam that needs a test with a directory in it.

    ``complete`` is false whenever the bench was interrupted or a case never ran: a partial record
    must never pass for a whole one.
    """

    bench_id: str
    site: SiteIdentity
    engine: EngineStamp
    corpus: CorpusIdentity = field(default_factory=CorpusIdentity)
    harness_hash: str | None = None
    # The ablation as data, quoted verbatim into the record — what was composed in and what was
    # taken out. The builder never interprets it (the harness spec is the runner's type), so a dial
    # added tomorrow lands in the record without an edit here.
    harness_dials: Mapping[str, object] | None = None
    runs: Sequence[CaseRun] = ()
    label: str | None = None
    started_utc: str | None = None
    finished_utc: str | None = None
    complete: bool = False
    # The table the dollars come from. ``None`` means the shipped one, exactly as
    # :func:`~noctis.eval.metrics.cost_metrics` reads it — and whichever it is, its version label
    # travels into both the record and the key.
    price_table: PriceTable | None = None
    # The site's failure vocabulary. ``None`` — or a taxonomy with no vocabulary for this site —
    # reports a ``null`` breakdown: counting failures into classes nobody declared would publish
    # zeros that look like an absence of failure.
    taxonomy: FailureTaxonomy | None = None
    configs: Sequence[ModelConfig] = ()
    client_stack: Mapping[str, str | None] | None = None


# ── the comparable key ─────────────────────────────────────────────────────────────────────


class ComparableKey(NamedTuple):
    """The seven-part identity two bench records must share before they may be subtracted.

    Rendered as one label — ``name=value`` pairs joined by :data:`KEY_SEPARATOR`, an unknown part
    spelled :data:`UNKNOWN` — so it is greppable component by component in an issue, a log, or a
    directory of records. Each part is here for a reason that has bitten a benchmark somewhere:
    the engine scores, the site version shapes what is asked, the prompt hash *is* what was asked,
    the corpus is the population, the harness is the composition under test, and the price table
    decides what a dollar figure means.
    """

    engine_version: int | None
    engine_fingerprint: str | None
    site_id: str | None
    site_version: str | None
    prompt_asset_hash: str | None
    corpus_version: str | None
    corpus_hash: str | None
    harness_hash: str | None
    pricing_table_version: str | None

    def __str__(self) -> str:
        """``engine_version=3|engine_fingerprint=…|site_id=coder|…`` — one greppable label."""
        return KEY_SEPARATOR.join(
            f"{name}={UNKNOWN if value is None else value}"
            for name, value in zip(self._fields, self, strict=True)
        )

    @property
    def complete(self) -> bool:
        """Whether every component is known — the precondition of any comparison at all."""
        return all(part is not None for part in self)


#: The key's components, in label order — the list a sensitivity check iterates.
KEY_COMPONENTS: tuple[str, ...] = ComparableKey._fields


def engine_digest(fingerprint: Mapping[str, str | None]) -> str | None:
    """Fold a per-component engine fingerprint into the one digest the key carries.

    **All components, not the arbiter pair.** The run record's key names ``gates`` and ``backtest``
    because those decide what a *scorecard* means; a bench measures an LLM site, whose result is
    shaped by the prompts it was assembled from and the gates that scored its output alike. So the
    fold is deliberately conservative: any component moving moves the bench key, which
    over-partitions comparison buckets slightly and never under-partitions them — the same trade
    ``observability/engine_id.py`` makes, and for the same reason (under-partitioning is the
    failure that silently corrupts results).

    All-or-none on a missing input, the fingerprint module's own rule: one ``None`` component makes
    the fold ``None``, because a digest over *some* of the engine looks like an identity while
    meaning less than one.
    """
    if not fingerprint:
        return None
    hasher = hashlib.sha256()
    for name in sorted(fingerprint):
        digest = fingerprint[name]
        if digest is None:
            return None
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(digest.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()[:_DIGEST_CHARS]


def comparable_key(artifacts: BenchArtifacts) -> ComparableKey:
    """The comparable key of one bench run, composed **verbatim** from its declared identities.

    Nothing here is inferred: each component is a value the runner resolved, folded (in the engine
    fingerprint's case) by a stated rule. The pricing table's version comes from the table itself,
    so the label can never disagree with the dollars the record publishes.
    """
    return ComparableKey(
        engine_version=artifacts.engine.engine_version,
        engine_fingerprint=engine_digest(artifacts.engine.fingerprint),
        site_id=artifacts.site.site_id,
        site_version=artifacts.site.version,
        prompt_asset_hash=artifacts.site.prompt_asset_hash,
        corpus_version=artifacts.corpus.version,
        corpus_hash=artifacts.corpus.hash,
        harness_hash=artifacts.harness_hash,
        pricing_table_version=_table(artifacts).version,
    )


def key_of(record: Mapping[str, object]) -> str | None:
    """One record's key label, or ``None`` when it carries nothing readable as one."""
    label = record.get("comparable_key")
    return label if isinstance(label, str) and label else None


def comparable(a: Mapping[str, object], b: Mapping[str, object]) -> bool:
    """Whether two bench records may have their numbers compared at all.

    True only when both carry the **same, fully known** key. An unknown component makes a record
    incomparable even to a record whose label matches it character for character: two ``null``s are
    two absences of evidence, and treating them as agreement is how an unidentifiable engine
    quietly gets compared to another unidentifiable engine.
    """
    key_a, key_b = key_of(a), key_of(b)
    return key_a is not None and key_a == key_b and _complete(key_a)


@dataclass(frozen=True)
class ComparablePair:
    """Two records produced under one key, side by side — and the deltas that are then meaningful.

    ``deltas`` is ``b - a`` per headline figure, ``None`` whenever either side is unknown: a delta
    against an unmeasured figure is not zero, it is nothing.
    """

    key: str
    figures: Mapping[str, Mapping[str, float | None]]
    deltas: Mapping[str, float | None]

    @property
    def comparable(self) -> bool:
        """Always true — the type itself is the statement that a comparison is admissible."""
        return True

    @property
    def banner(self) -> str | None:
        """No banner: there is nothing to warn a reader about."""
        return None


@dataclass(frozen=True)
class IncomparablePair:
    """Two records that were **not** produced under one key: presentable, never subtractable.

    The refusal is structural, which is the whole point — this type carries no ``deltas`` field for
    a consumer to reach for, so "present the two numbers, do not subtract them" is enforced by what
    it was handed rather than by a convention it might ignore. ``differences`` names every key
    component that moved, so the operator's next question ("what changed?") is already answered.
    """

    banner: str
    key_a: str | None
    key_b: str | None
    differences: tuple[str, ...]
    figures: Mapping[str, Mapping[str, float | None]]

    @property
    def comparable(self) -> bool:
        return False


#: What :func:`side_by_side` returns — the admissible comparison, or the refusal.
Comparison: TypeAlias = ComparablePair | IncomparablePair


def side_by_side(a: Mapping[str, object], b: Mapping[str, object]) -> Comparison:
    """Put two bench records next to each other — with deltas, or with a banner and none.

    The one comparison primitive this story ships. Paired statistics over two benchmark runs (the
    bootstrap, the significance question, the per-case pairing) are their own epic's work; what is
    settled here is that they can never be asked of two records that were not comparable, because
    the incomparable case never produces a value with a delta in it.
    """
    figures = {"a": _figures(a), "b": _figures(b)}
    key_a, key_b = key_of(a), key_of(b)
    if comparable(a, b) and key_a is not None:
        deltas = _deltas(figures["a"], figures["b"])
        return ComparablePair(key=key_a, figures=figures, deltas=deltas)
    differences = _differences(key_a, key_b)
    return IncomparablePair(
        banner=_banner(key_a, key_b, differences),
        key_a=key_a,
        key_b=key_b,
        differences=differences,
        figures=figures,
    )


def _banner(key_a: str | None, key_b: str | None, differences: tuple[str, ...]) -> str:
    """The loud sentence a reader cannot miss, naming why no delta is on offer."""
    if differences:
        named = "; ".join(differences)
        return (
            f"INCOMPARABLE — these two bench records were produced under different conditions "
            f"({named}). Present them side by side; subtracting them would attribute the "
            f"difference between two arenas to the thing under test."
        )
    unknown = sorted(
        {
            name
            for label in (key_a, key_b)
            for name, value in _components(label).items()
            if value == UNKNOWN
        }
    )
    if unknown:
        return (
            "INCOMPARABLE — the comparable key carries unknown component(s): "
            f"{', '.join(unknown)}. An unknown component is not evidence of sameness, so these "
            "two records may be read side by side but never subtracted."
        )
    return (
        "INCOMPARABLE — at least one of these records carries no comparable key at all, so nothing "
        "is known about the conditions behind its numbers."
    )


def _complete(label: str) -> bool:
    """Whether a key label names every component and leaves none of them unknown."""
    parsed = _components(label)
    return set(parsed) == set(KEY_COMPONENTS) and UNKNOWN not in parsed.values()


def _components(label: str | None) -> dict[str, str]:
    """One key label, read back as its ``name=value`` components."""
    parsed: dict[str, str] = {}
    for part in (label or "").split(KEY_SEPARATOR):
        name, separator, value = part.partition("=")
        if separator:
            parsed[name] = value
    return parsed


def _differences(key_a: str | None, key_b: str | None) -> tuple[str, ...]:
    """Every key component the two records disagree about, named once, with both values.

    Declared components first, in key order, then anything else either label carries — a record
    written by a later Noctis whose key grew a component is still diffable by this one, and the
    component it does not know about is reported rather than dropped.
    """
    left, right = _components(key_a), _components(key_b)
    later = sorted((set(left) | set(right)) - set(KEY_COMPONENTS))
    names = list(KEY_COMPONENTS) + later
    return tuple(
        f"{name}: {left.get(name, 'absent')} vs {right.get(name, 'absent')}"
        for name in names
        if left.get(name) != right.get(name)
    )


def _figures(record: Mapping[str, object]) -> dict[str, float | None]:
    """The headline numbers of one record, ``None`` for anything it does not carry."""
    metrics = record.get("metrics")
    section: Mapping[str, object] = metrics if isinstance(metrics, Mapping) else {}
    cost = section.get("cost")
    latency = section.get("latency")
    flat: dict[str, object] = {
        **{name: value for name, value in section.items() if not isinstance(value, Mapping)},
        **(cost if isinstance(cost, Mapping) else {}),
        **(latency if isinstance(latency, Mapping) else {}),
    }
    return {name: _number(flat.get(name)) for name in HEADLINE_FIGURES}


def _deltas(
    left: Mapping[str, float | None], right: Mapping[str, float | None]
) -> dict[str, float | None]:
    """``b - a`` per headline figure — ``None`` the moment either side is unknown."""
    return {
        name: (
            None
            if left.get(name) is None or right.get(name) is None
            else round(float(right[name] or 0.0) - float(left[name] or 0.0), 6)
        )
        for name in HEADLINE_FIGURES
    }


def _number(value: object) -> float | None:
    """One record value as a number, or ``None`` for anything that is not one."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


# ── the builder ────────────────────────────────────────────────────────────────────────────


def build(artifacts: BenchArtifacts) -> dict:
    """Assemble the bench record. Pure: same artifacts in, byte-identical document out."""
    results = tuple(run.result for run in artifacts.runs)
    metrics = compute_metrics(results, table=artifacts.price_table)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        # Composed verbatim from the identities below, and stored so a consumer never has to
        # recompose it (and so two records can be bucketed by a string comparison).
        "comparable_key": str(comparable_key(artifacts)),
        "bench": {
            "bench_id": artifacts.bench_id,
            "label": artifacts.label,
            "started_utc": artifacts.started_utc,
            "finished_utc": artifacts.finished_utc,
            "complete": bool(artifacts.complete),
            # Derived from the same metrics call the block below is rendered from, so the counts a
            # reader sees at the top can never disagree with the rates underneath them.
            "cases": metrics.cases,
            "runs": metrics.runs,
            "attempts": sum(len(result.attempts) for result in results),
        },
        "site": {
            "site_id": artifacts.site.site_id,
            "site_version": artifacts.site.version,
            "prompt_asset_hash": artifacts.site.prompt_asset_hash,
            "prompt_asset_groups": list(artifacts.site.prompt_asset_groups),
        },
        "corpus": {
            "version": artifacts.corpus.version,
            "hash": artifacts.corpus.hash,
            "case_count": artifacts.corpus.case_count,
            "split": artifacts.corpus.split,
        },
        "harness": {
            "hash": artifacts.harness_hash,
            "dials": dict(artifacts.harness_dials) if artifacts.harness_dials is not None else None,
        },
        "engine": {
            "engine_version": artifacts.engine.engine_version,
            "noctis_version": artifacts.engine.noctis_version,
            "fingerprint": dict(artifacts.engine.fingerprint),
            "fingerprint_digest": engine_digest(artifacts.engine.fingerprint),
        },
        "metrics": _metrics(metrics),
        "failures": _failures(artifacts, results),
        # The per-case evidence every aggregate above was derived from — carried in full, including
        # each attempt's usage split, so a reader can recompute the whole metrics block rather than
        # take it on trust.
        "results": [_result(run) for run in artifacts.runs],
        "provenance": _provenance(artifacts, metrics),
    }


def _metrics(metrics: BenchMetrics) -> dict:
    """The metrics module's record, rendered — its arithmetic, this record's vocabulary.

    Two renamings and nothing else: the mapping keys become strings (JSON has no integer keys) and
    the latency fields take the record's canonical ``_s`` suffix. Every number is the value
    :func:`~noctis.eval.metrics.compute_metrics` returned, ``None`` included.
    """
    distribution = metrics.attempt_distribution
    return {
        "first_attempt_pass_rate": metrics.first_attempt_pass_rate,
        "job_pass_rate": metrics.job_pass_rate,
        "mean_attempts_to_pass": metrics.mean_attempts_to_pass,
        "attempt_distribution": (
            None
            if distribution is None
            else {
                "passed_on": {str(index): share for index, share in distribution.passed_on.items()},
                "never": distribution.never,
            }
        ),
        "retry_yield": (
            None
            if metrics.retry_yield is None
            else {str(index): share for index, share in metrics.retry_yield.items()}
        ),
        "cost": {
            "tokens_total": metrics.cost.tokens_total,
            "tokens_per_pass": metrics.cost.tokens_per_pass,
            "usd_total_estimate": metrics.cost.usd_total_estimate,
            "usd_per_pass_estimate": metrics.cost.usd_per_pass_estimate,
            "pricing_table_version": metrics.cost.pricing_table_version,
        },
        "latency": {"p50_s": metrics.latency.p50_seconds, "p95_s": metrics.latency.p95_seconds},
    }


def _failures(artifacts: BenchArtifacts, results: Sequence[CaseResult]) -> dict | None:
    """How the failures broke down, in the site's own vocabulary — or ``null`` when there is none.

    ``null`` rather than zeros for a bench with no registered vocabulary, because a breakdown of
    zeros reads as "nothing failed" about a bench nobody classified. ``failed_attempts`` sits beside
    the breakdown's ``total`` so the gap between them — failures that recorded no error text — is
    visible rather than silently absent from the denominators.
    """
    taxonomy = artifacts.taxonomy
    if taxonomy is None:
        return None
    failed = [attempt for result in results for attempt in result.attempts if not attempt.passed]
    errors = [attempt.error for attempt in failed if attempt.error is not None]
    try:
        breakdown = taxonomy.classify(artifacts.site.site_id, errors)
    except UnknownVocabulary:
        return None
    return {
        "counts": dict(breakdown.counts),
        "shares": dict(breakdown.shares),
        "total": breakdown.total,
        # The rot gauge, readable on its own: a share that climbs run over run says the matchers
        # have stopped covering the failures they were written for.
        "unclassified_share": breakdown.shares[UNCLASSIFIED],
        "failed_attempts": len(failed),
    }


def _result(run: CaseRun) -> dict:
    """One rep, with the per-attempt evidence and the per-rep facts derived from it."""
    result = run.result
    return {
        "case_id": result.case_id,
        "config_id": run.config_id,
        "ran": result.ran,
        "first_attempt_passed": result.first_attempt_passed,
        "passed": result.passed,
        "attempts_to_pass": result.attempts_to_pass,
        "latency_s": result.seconds_to_pass,
        "attempts": [
            _attempt(attempt, run.served(index)) for index, attempt in enumerate(result.attempts)
        ],
    }


def _attempt(attempt: AttemptOutcome, served_model: str | None) -> dict:
    """One attempt: what it did, what it spent, how long it took, and what actually served it.

    The usage split is carried field by field — ``null`` where the seam reported nothing — because
    a total alone cannot be re-priced, and a record whose bill nobody can recompute is a bill
    nobody can check.
    """
    usage = attempt.usage()
    return {
        "passed": attempt.passed,
        "model": attempt.model,
        # Provenance tier: recorded here, never in the key. ``null`` when the provider reported no
        # served id — absence is never back-filled from the alias that was requested.
        "served_model": served_model,
        "tokens": attempt.tokens,
        "usage": {name: usage.get(name) for name in USAGE_FIELDS},
        "latency_s": attempt.seconds,
        "error": attempt.error,
    }


def _provenance(artifacts: BenchArtifacts, metrics: BenchMetrics) -> dict:
    """The tier that is recorded and never keyed: who was asked, what answered, on which stack."""
    return {
        "configs": [_config(config, artifacts) for config in artifacts.configs],
        "client_stack": (
            dict(artifacts.client_stack) if artifacts.client_stack is not None else None
        ),
        # The same label the key carries and the cost figures were produced under — one source.
        "pricing_table_version": metrics.cost.pricing_table_version,
    }


def _config(config: ModelConfig, artifacts: BenchArtifacts) -> dict:
    """One requested configuration, and every served model id observed under it.

    ``alias_drift`` is **derived**, never asserted: true when more than one distinct served id
    answered this alias, false when exactly one did, and ``null`` when none was ever recorded —
    "no drift" would be a claim about evidence that does not exist.
    """
    served: list[str] = []
    attempts = 0
    unrecorded = 0
    for run in artifacts.runs:
        if run.config_id != config.config_id:
            continue
        for index in range(len(run.result.attempts)):
            attempts += 1
            model = run.served(index)
            if model is None:
                unrecorded += 1
            elif model not in served:
                served.append(model)
    return {
        "config_id": config.config_id,
        "provider": config.provider,
        "requested_model": config.requested_model,
        "served_models": sorted(served),
        "served_unrecorded": unrecorded,
        "alias_drift": None if not served else len(served) > 1,
        "attempts": attempts,
    }


def _table(artifacts: BenchArtifacts) -> PriceTable:
    """The price table these numbers were produced under — the shipped one by default.

    Read exactly the way :func:`~noctis.eval.metrics.cost_metrics` reads it, so the version in the
    key, the version in the cost block and the prices behind the dollars are one decision.
    """
    return artifacts.price_table if artifacts.price_table is not None else default_table()


# ── the validator ──────────────────────────────────────────────────────────────────────────

_BENCH_KEYS = (
    "bench_id",
    "label",
    "started_utc",
    "finished_utc",
    "complete",
    "cases",
    "runs",
    "attempts",
)
_SITE_KEYS = ("site_id", "site_version", "prompt_asset_hash", "prompt_asset_groups")
_CORPUS_KEYS = ("version", "hash", "case_count", "split")
_HARNESS_KEYS = ("hash", "dials")
_ENGINE_KEYS = ("engine_version", "noctis_version", "fingerprint", "fingerprint_digest")
_METRICS_KEYS = (
    "first_attempt_pass_rate",
    "job_pass_rate",
    "mean_attempts_to_pass",
    "attempt_distribution",
    "retry_yield",
    "cost",
    "latency",
)
_COST_KEYS = (
    "tokens_total",
    "tokens_per_pass",
    "usd_total_estimate",
    "usd_per_pass_estimate",
    "pricing_table_version",
)
_LATENCY_KEYS = ("p50_s", "p95_s")
_FAILURE_KEYS = ("counts", "shares", "total", "unclassified_share", "failed_attempts")
_RESULT_KEYS = (
    "case_id",
    "config_id",
    "ran",
    "first_attempt_passed",
    "passed",
    "attempts_to_pass",
    "latency_s",
    "attempts",
)
_ATTEMPT_KEYS = ("passed", "model", "served_model", "tokens", "usage", "latency_s", "error")
_PROVENANCE_KEYS = ("configs", "client_stack", "pricing_table_version")
_CONFIG_KEYS = (
    "config_id",
    "provider",
    "requested_model",
    "served_models",
    "served_unrecorded",
    "alias_drift",
    "attempts",
)

# The subtrees the record *quotes* rather than authors, and therefore does not impose its own unit
# vocabulary on: the harness dials are the harness spec's own names, and the client stack is a
# package index's. Renaming a key inside either would make the record disagree with the thing it
# claims to quote — the run record draws exactly this line around its frozen settings.
_VERBATIM_SUBTREES = ("harness.dials", "provenance.client_stack")


def validate(record: Mapping[str, object]) -> list[str]:
    """Check one bench record against the schema; return **every** problem, one line each.

    A list, not an exception, for the run record's reason: the operator asking "does this record
    conform?" wants the whole list at once, not one problem per attempt. An undeclared key is one
    of the answers — the walkers are the run record's own, and they check a block's vocabulary as
    well as its presence. Nothing here reads a file or a clock, so a record checks the same in
    memory as it does on disk.
    """
    problems: list[str] = []
    if record.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version: expected {SCHEMA_VERSION}, found {record.get('schema_version')!r}"
        )
    if record.get("kind") != KIND:
        problems.append(f"kind: expected {KIND!r}, found {record.get('kind')!r}")
    for section in REQUIRED_SECTIONS:
        if section not in record:
            problems.append(f"{section}: section is missing")

    problems += _check_key(record.get("comparable_key"))
    problems += _check_section("bench", record.get("bench"), _BENCH_KEYS)
    problems += _check_section("site", record.get("site"), _SITE_KEYS)
    problems += _check_section("corpus", record.get("corpus"), _CORPUS_KEYS)
    problems += _check_section("harness", record.get("harness"), _HARNESS_KEYS)
    problems += _check_section("engine", record.get("engine"), _ENGINE_KEYS)
    problems += _check_metrics(record.get("metrics"))
    problems += _check_section("failures", record.get("failures"), _FAILURE_KEYS)
    problems += _check_results(record.get("results"))
    problems += _check_provenance(record.get("provenance"))
    # The document-wide conventions, checked over the whole record rather than section by section
    # (so a section added tomorrow inherits them) and by the run record's **own** walkers rather
    # than by a second copy of them, so the two documents can never drift apart on what ``_s``
    # means or on what a dollar figure has to call itself.
    problems += check_units("", record, verbatim=_VERBATIM_SUBTREES)
    problems += check_stamps("", record)
    problems += check_estimate_labels("", record, verbatim=_VERBATIM_SUBTREES)
    return problems


def _check_key(label: object) -> list[str]:
    """The key is one string naming every component, in order — the bucket a reader greps by."""
    if label is None:
        return []
    if not isinstance(label, str):
        return ["comparable_key: must be a string"]
    named = tuple(_components(label))
    if named != KEY_COMPONENTS:
        return [
            "comparable_key: names "
            f"{', '.join(named) or 'nothing'} — a key states every component in order: "
            f"{', '.join(KEY_COMPONENTS)}"
        ]
    return []


def _check_metrics(metrics: object) -> list[str]:
    """The metrics block and its two sub-blocks — presence is the whole contract, as above."""
    problems = _check_section("metrics", metrics, _METRICS_KEYS)
    if not isinstance(metrics, Mapping):
        return problems
    problems += _check_section("metrics.cost", metrics.get("cost"), _COST_KEYS)
    return problems + _check_section("metrics.latency", metrics.get("latency"), _LATENCY_KEYS)


def _check_results(results: object) -> list[str]:
    """The per-case evidence: one entry per rep, each carrying its attempts in order."""
    if results is None:
        return []
    if not isinstance(results, Sequence) or isinstance(results, str | bytes):
        return ["results: must be a list"]
    problems: list[str] = []
    for position, result in enumerate(results):
        label = f"results[{position}]"
        if not isinstance(result, Mapping):
            problems.append(f"{label}: must be an object")
            continue
        problems += check_keys(label, result, _RESULT_KEYS)
        attempts = result.get("attempts")
        if (
            attempts is None
            or isinstance(attempts, str | bytes)
            or not isinstance(attempts, Sequence)
        ):
            problems.append(f"{label}.attempts: must be a list")
            continue
        for index, attempt in enumerate(attempts):
            entry = f"{label}.attempts[{index}]"
            if not isinstance(attempt, Mapping):
                problems.append(f"{entry}: must be an object")
                continue
            problems += check_keys(entry, attempt, _ATTEMPT_KEYS)
    return problems


def _check_provenance(provenance: object) -> list[str]:
    """The recorded-not-keyed tier: every config states what was asked and what answered."""
    problems = _check_section("provenance", provenance, _PROVENANCE_KEYS)
    if not isinstance(provenance, Mapping):
        return problems
    configs = provenance.get("configs")
    if configs is None:
        return problems
    if not isinstance(configs, Sequence) or isinstance(configs, str | bytes):
        return [*problems, "provenance.configs: must be a list"]
    for position, config in enumerate(configs):
        label = f"provenance.configs[{position}]"
        if not isinstance(config, Mapping):
            problems.append(f"{label}: must be an object")
            continue
        problems += check_keys(label, config, _CONFIG_KEYS)
    return problems


def _check_section(label: str, section: object, keys: Sequence[str]) -> list[str]:
    """One section: ``null``, or an object carrying every key the contract names and no other."""
    if section is None:
        return []
    if not isinstance(section, Mapping):
        return [f"{label}: section must be an object or null"]
    return check_keys(label, section, keys)
