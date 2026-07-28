"""The run record's schema — its version, its caps, and a pure validator (story #129, epic #126).

One run, one self-describing file: ``workspace/runs/<run_id>/run.json``. This module holds the
contract that file promises to anything reading it (a website, a later Noctis, an operator with
``jq``), and nothing else — no I/O, no clock, no config, not even the record's construction. It is
deliberately the smallest module in the epic, because it is also the one the engine fingerprint
tracks as the ``schema`` component (``observability/engine_id.py``): changing what is recorded
should move a digest, and only that.

**Versioning is additive-only.** :data:`SCHEMA_VERSION` is 1. New fields may be added at any time;
an existing field never changes meaning or type, and a reader ignores keys it does not know. That
is what lets a record written tonight still be read by the Noctis that resumes the run in a month.
A breaking change bumps the version and upgrades on read — it never silently repurposes a key.

**Two conventions are part of the contract, not style.** Units are explicit in the field name
(``_usd``, ``_pct``, ``_bps``, ``_s``, ``_bytes``), and a known-absent value is an explicit
``null`` rather than an omitted key — so a consumer can tell "not applicable" from "this schema
version did not have it". :func:`validate` enforces both where it can.

**Caps are honest.** Bulky lists are bounded (:data:`TRADE_CAP`, :data:`EVENT_CAP`) and every cap
that bites writes a ``truncated`` note carrying kept/total counts. Silent truncation is forbidden:
a truncated record must never pass for a complete one. Segments and sessions are deliberately
uncapped — they are the run's spine, and losing one would make every derived total a lie.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = [
    "EVENT_CAP",
    "KIND",
    "REQUIRED_SECTIONS",
    "RUN_STATUSES",
    "SCHEMA_VERSION",
    "SEGMENT_COMMANDS",
    "SEGMENT_STATUSES",
    "TRADE_CAP",
    "validate",
]

# The record contract's version. Additive-only: bump this only for a breaking change.
SCHEMA_VERSION = 1

# The record's self-declared type, so a consumer can tell a run record from any other JSON.
KIND = "noctis.run"

# The top-level sections every record carries. Later stories in the epic add sections
# (``research``, ``strategies``, ``sessions``, ``assumptions``); additive-only means this set may
# grow, never shrink. ``performance`` is here from story #137 as an explicit ``null``: a run that
# never traded must report *no* performance rather than zeros, and a consumer has to be able to
# tell "this run is still researching" from "this schema version had no such key".
REQUIRED_SECTIONS = (
    "run",
    "segments",
    # What the run cost and what that bought (story #140). ``null`` for a run with no research
    # evidence at all — the shape of a run with no LLM key, which must report an unknown bill
    # rather than a free one.
    "spend",
    # The newest machine this run has worked on (story #139), derived from the per-segment blocks
    # beside it. ``null`` for a run no segment of which measured an environment.
    "environment_latest",
    "engine",
    "inputs",
    "performance",
    "events",
    "errors",
)

# The run's lifecycle. ``interrupted`` is observed on the next open (a segment with a start stamp
# and no stop stamp), never guessed at write time. ``completed`` is terminal.
RUN_STATUSES = ("running", "stopped", "interrupted", "completed")

# One segment = one process invocation. It is open (``running``), closed cleanly (``stopped``), or
# was killed mid-flight and found that way on the next open (``interrupted``).
SEGMENT_STATUSES = ("running", "stopped", "interrupted")

# The verbs that append a segment, and therefore the only two kinds of segment a run has: the
# day/night loop (``run``) and a standalone observable research session (``research``, story #137).
# A **research-only segment is derived from this**, not marked by a second flag — ``research`` is a
# verb that cannot trade, so "which segments were research-only" is answerable from what the record
# already had to carry: the command each process was invoked as.
SEGMENT_COMMANDS = ("run", "research")

# Bounds on the two lists that grow without limit in a long run. A record is meant to be fetched
# whole by a website, so these keep a multi-week run in the tens of kilobytes.
TRADE_CAP = 5_000
EVENT_CAP = 2_000

# The keys the ``run`` section always carries — presence is the contract, ``null`` is a value.
_RUN_KEYS = (
    "run_id",
    "label",
    "status",
    "created_utc",
    "last_active_utc",
    "completed_utc",
    # The run-level compute cap in hours (story #136), ``null`` when uncapped. Frozen at creation
    # with the rest of the configuration and surfaced here so two runs can be compared on equal
    # compute — 100 research hours and 30 are not the same experiment.
    "run_limit_hours",
    # Whether this run ever placed a paper order (story #137). ``false`` ⇒ ``performance`` is
    # ``null`` throughout, because a run that never traded has no equity curve — and a rendered
    # flat 0% one would be a lie about a run that is simply still researching.
    "traded",
    # The three cumulative totals, every one of them **derived** from ``segments[]`` at write time
    # and never incremented in memory (epic D4). The two phase totals are ``null`` — not ``0`` —
    # when no segment measured a phase at all.
    "cumulative_runtime_s",
    "cumulative_research_s",
    "cumulative_trading_s",
    # Trials this run journaled, counted off its own ``state/experiments/*.jsonl`` at write time —
    # the same lines the exhaustion gate counts, and the multiple-testing count a deflated Sharpe
    # needs. ``null`` when the run journaled nothing at all.
    "cumulative_trials",
    # Whether retention removed this run's heavy directories (story #138). ``false`` on every run
    # that was never pruned; ``true`` says ``state/``, ``strategies/`` and ``reports/`` are gone, so
    # the record's path-plus-hash references into them no longer resolve — everything the record
    # *embeds* is still here, because the record is never pruned. Only a ``completed`` run can carry
    # it: pruning a resumable run's state would destroy its resumability, so it is refused.
    "state_pruned",
    "complete",
    "truncated",
)

_SEGMENT_KEYS = (
    "index",
    "started_utc",
    "stopped_utc",
    "duration_s",
    "stopped_reason",
    "status",
    "argv",
    "command",
    "resumed",
    "counters",
    # Seconds this segment spent working in each phase (``{"RESEARCH": …, "TRADING": …}``), or
    # ``null`` when it measured none. The per-segment fact the run's cumulative research/trading
    # seconds are summed from — attributed to the process that produced it, like ``counters``.
    "phase_seconds",
    # The machine this segment ran on and the inputs it resolved (story #139), or ``null`` when it
    # measured none. **Per segment**, because a run may migrate machines mid-experiment and
    # research throughput is CPU-bound: one night's trials-per-hour is only comparable to
    # another's when the hardware behind each is on the record.
    "environment",
    "engine_version",
    "engine_fingerprint",
)

# The environment block's own keys. Every one is present whenever the block is (an absent *fact*
# is an explicit ``null`` plus the missing capability named in ``degraded_seams``), so a reader can
# tell "this machine had no git checkout" from "this schema version did not record git".
_ENVIRONMENT_KEYS = (
    # ``sha256(hostname)[:12]`` — comparable across segments, never a machine name (story #129).
    "hostname_hash",
    "os",
    "container",
    "cpu",
    "memory_total_bytes",
    "disk_free_bytes",
    "python",
    "noctis_version",
    "git",
    "lockfile_digest",
    # ``{extra name: version or null}`` over the optional extras the installer knows, and the
    # names of every capability that was missing — the ``psutil`` hardware extra included, since it
    # is an extra rather than a core dependency by design.
    "extras_present",
    "degraded_seams",
)

_ENGINE_KEYS = (
    "engine_version",
    "engine_epoch",
    "noctis_version",
    "fingerprint",
    "comparable_key",
    "mixed_engine",
    "engine_changes",
)

# One deliberately accepted engine change (story #135, ``--allow-engine-upgrade``). The twin of
# ``inputs.config_changes``, in the section whose epoch it bumps and with the same rule: a run
# whose engine changed mid-flight says so *and says where*, naming every component that moved and
# both its digests. Empty (never absent) on the runs that never upgraded.
_ENGINE_CHANGE_KEYS = (
    "at",
    "segment",
    "from_epoch",
    "to_epoch",
    "from_engine_version",
    "to_engine_version",
    "components",
    "accepted_by",
)

_EVENT_KEYS = ("t", "segment", "kind", "text")

# The run's frozen configuration (story #132). Present as an explicit ``null`` on a run that never
# froze one — an adopted history — and otherwise carrying the rehydration source plus the tier
# lists that say which keys a resume restores and which it takes from the live process.
_INPUTS_KEYS = (
    "config_epoch",
    "config_changes",
    "frozen_at_utc",
    "execution_mode",
    "mandate",
    # The rest of the provenance block (story #139): which models this run researches, authors,
    # escalates and ideates with, and which vendor/dataset its bars came from. Both are derived
    # views over values ``settings.resolved`` already carries — stated once, resolved, so a
    # consumer never has to reconstruct a fallback chain to know what produced these numbers.
    "models",
    "data",
    "settings",
)
_INPUT_MODEL_KEYS = (
    "research",
    "coder",
    "coder_fallback",
    "ideation",
    "research_loop",
    "context_window",
    "cost_profile",
)
_INPUT_DATA_KEYS = ("provider", "dataset", "lake_dir")
_INPUT_SETTINGS_KEYS = ("digest", "resolved", "frozen_keys", "live_keys", "refused_keys")

# One deliberate mid-run config change (story #134, ``--rebase-config``). The list lives **inside
# ``inputs``**, beside the ``config_epoch`` each entry bumps, rather than in a per-segment list:
# the run's configuration is one run-level thing, and every entry names the ``segment`` it happened
# in — the same shape the record's one event stream already uses, so a reader gets a single
# chronology instead of stitching per-segment lists together. Empty (never absent) on the runs that
# never rebased, which is how "this run's config never changed" is a stated fact.
_CONFIG_CHANGE_KEYS = (
    "at",
    "segment",
    "from_epoch",
    "to_epoch",
    "digest_before",
    "digest_after",
    "settings",
    "mandate",
)

# The spend block (story #140): the token split three ways, what it cost, which price table said
# so, and the efficiency ratios the whole thing exists to make comparable. Present as an explicit
# ``null`` for a run with no research evidence, and otherwise carrying every key — with ``null``
# wherever a number is genuinely unknown (an unpriced model, a ledger with no split, a ratio with a
# zero denominator).
_SPEND_KEYS = (
    "tokens",
    "by_model",
    "by_stage",
    "by_segment",
    "llm_usd_estimate",
    "pricing_table_version",
    "efficiency",
)
_EFFICIENCY_KEYS = (
    "usd_per_champion_estimate",
    "usd_per_trial_estimate",
    "trials_per_hour",
    "research_hours_per_champion",
)

# The marker every dollar figure in the record must carry. Prices come from a versioned list-price
# table and are therefore *estimates*; a field that named itself ``usd`` alone would read as a
# receipt, and the difference matters most to exactly the reader who is comparing two runs on cost.
USD_MARKER = "usd"
ESTIMATE_MARKER = "estimate"

# The two settings a record may never carry, whatever else it grows: the live-money double gate.
# The safety gate re-resolves from the config file and the ALLOW_LIVE environment variable at every
# process start, so a record that carried either one could offer a second source for a decision
# that must have exactly two independent ones (AGENTS.md rule 1).
UNRECORDABLE_SETTINGS = ("mode", "allow_live")


def validate(record: Mapping[str, object]) -> list[str]:
    """Check one record against the schema; return the problems, one line each (``[]`` = valid).

    A **list, not an exception**: a validator that raises reports one problem per run, and the
    caller who most needs this — an operator asking "is this record readable?" — wants the whole
    list at once. Nothing here reads a file or a clock, so a record can be checked in memory
    exactly as it will be checked on disk.
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

    run = record.get("run")
    if isinstance(run, Mapping):
        problems += _check_keys("run", run, _RUN_KEYS)
        problems += _check_status("run.status", run.get("status"), RUN_STATUSES)
        problems += _check_stamp("run.created_utc", run.get("created_utc"))
        problems += _check_stamp("run.last_active_utc", run.get("last_active_utc"))
        problems += _check_stamp("run.completed_utc", run.get("completed_utc"))
    elif "run" in record:
        problems.append("run: section must be an object")

    problems += _check_segments(record.get("segments"))
    problems += _check_environment("environment_latest", record.get("environment_latest"))

    engine = record.get("engine")
    if isinstance(engine, Mapping):
        problems += _check_keys("engine", engine, _ENGINE_KEYS)
        problems += _check_changes(
            "engine.engine_changes", engine.get("engine_changes"), _ENGINE_CHANGE_KEYS
        )
    elif "engine" in record:
        problems.append("engine: section must be an object")

    problems += _check_inputs(record.get("inputs"))
    problems += _check_spend(record.get("spend"))
    problems += _check_performance(record.get("performance"), run=run)

    for name in ("events", "errors"):
        problems += _check_events(name, record.get(name))
    return problems


def _check_performance(performance: object, *, run: object) -> list[str]:
    """A run that never traded reports **no** performance — ``null``, never zeros (epic D10, §5.6).

    The whole rule, enforced structurally so it cannot be lost when story #142 fills the block in:
    a research-only run (many research segments, no trades) is a *first-class* shape, and a
    consumer rendering it must be able to say "researching" rather than plot a flat 0% equity curve
    it was handed as if it were a result. So the pairing is part of the contract — ``traded: false``
    and a populated ``performance`` is a schema violation, not a rounding detail.
    """
    if performance is None:
        return []
    if not isinstance(performance, Mapping):
        return ["performance: section must be an object or null"]
    traded = run.get("traded") if isinstance(run, Mapping) else None
    if traded is False:
        return [
            "performance: must be null when run.traded is false — a run that never traded has no "
            "realised performance, and zeros would render as a result it never produced"
        ]
    return []


def _check_spend(spend: object) -> list[str]:
    """The spend block: ``null``, or every key present — and every cost field an *estimate*.

    The labelling check is structural rather than a convention, because the failure it prevents is
    silent: a website rendering ``usd`` as a charge, when the number came from a list-price table
    that ignores discounts, batch tiers and mid-month changes. So a dollar-bearing key that does
    not say ``estimate`` is a schema violation, wherever in the block it appears.
    """
    if spend is None:
        return []
    if not isinstance(spend, Mapping):
        return ["spend: section must be an object or null"]
    problems = _check_keys("spend", spend, _SPEND_KEYS)
    problems += _check_block("spend.efficiency", spend.get("efficiency"), _EFFICIENCY_KEYS)
    return problems + _check_estimate_labels("spend", spend)


def _check_estimate_labels(label: str, section: Mapping[str, object]) -> list[str]:
    """Every key naming dollars, at any depth, must also name itself an estimate."""
    problems: list[str] = []
    for key, value in section.items():
        name = str(key)
        if USD_MARKER in name.lower() and ESTIMATE_MARKER not in name.lower():
            problems.append(
                f"{label}.{name}: a cost field must name itself an estimate "
                f"(e.g. {name}_{ESTIMATE_MARKER}) — these prices come from a versioned table, "
                "not from an invoice"
            )
        if isinstance(value, Mapping):
            problems += _check_estimate_labels(f"{label}.{name}", value)
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            for position, item in enumerate(value):
                if isinstance(item, Mapping):
                    problems += _check_estimate_labels(f"{label}.{name}[{position}]", item)
    return problems


def _check_inputs(inputs: object) -> list[str]:
    """The frozen inputs: absent as an explicit ``null``, and never carrying the two gates."""
    if inputs is None:
        return []
    if not isinstance(inputs, Mapping):
        return ["inputs: section must be an object or null"]
    problems = _check_keys("inputs", inputs, _INPUTS_KEYS)
    problems += _check_stamp("inputs.frozen_at_utc", inputs.get("frozen_at_utc"))
    problems += _check_config_changes(inputs.get("config_changes"))
    problems += _check_block("inputs.models", inputs.get("models"), _INPUT_MODEL_KEYS)
    problems += _check_block("inputs.data", inputs.get("data"), _INPUT_DATA_KEYS)
    settings = inputs.get("settings")
    if not isinstance(settings, Mapping):
        return [*problems, "inputs.settings: must be an object"]
    problems += _check_keys("inputs.settings", settings, _INPUT_SETTINGS_KEYS)
    resolved = settings.get("resolved")
    if isinstance(resolved, Mapping):
        problems += [
            f"inputs.settings.resolved.{key}: the live-money gates are never recorded"
            for key in UNRECORDABLE_SETTINGS
            if key in resolved
        ]
    return problems


def _check_config_changes(changes: object) -> list[str]:
    """Every deliberate config change carries its before/after, its stamp, and its segment.

    A mid-run config change is never silent (epic §5.4): a record that changed configuration must
    say so *and say where*, or every comparison built on it is false. So the segment index is part
    of the contract, not an optional nicety.
    """
    return _check_changes("inputs.config_changes", changes, _CONFIG_CHANGE_KEYS)


def _check_changes(label: str, changes: object, keys: Sequence[str]) -> list[str]:
    """One deliberate-mid-run-change list — the config's and the engine's, checked identically.

    The two are the same contract at two levels (what the run was told, and what judged it), so
    they get one implementation: a list of objects, each carrying its keys, its UTC stamp and the
    segment it happened in. Absent lists are fine; a list of anything else is not.
    """
    if changes is None:
        return []
    if not isinstance(changes, Sequence) or isinstance(changes, str | bytes):
        return [f"{label}: must be a list"]
    problems: list[str] = []
    for position, change in enumerate(changes):
        entry = f"{label}[{position}]"
        if not isinstance(change, Mapping):
            problems.append(f"{entry}: must be an object")
            continue
        problems += _check_keys(entry, change, keys)
        problems += _check_stamp(f"{entry}.at", change.get("at"))
    return problems


def _check_segments(segments: object) -> list[str]:
    """Segments are an append-only sequence, indexed from 0 with no gaps — the run's spine."""
    if segments is None:
        return []
    if not isinstance(segments, Sequence) or isinstance(segments, str | bytes):
        return ["segments: must be a list"]
    problems: list[str] = []
    for position, segment in enumerate(segments):
        label = f"segments[{position}]"
        if not isinstance(segment, Mapping):
            problems.append(f"{label}: must be an object")
            continue
        problems += _check_keys(label, segment, _SEGMENT_KEYS)
        if segment.get("index") != position:
            problems.append(
                f"{label}.index: expected {position}, found {segment.get('index')!r} "
                "(segments are append-only and indexed from 0)"
            )
        problems += _check_status(f"{label}.status", segment.get("status"), SEGMENT_STATUSES)
        problems += _check_status(f"{label}.command", segment.get("command"), SEGMENT_COMMANDS)
        problems += _check_stamp(f"{label}.started_utc", segment.get("started_utc"))
        problems += _check_stamp(f"{label}.stopped_utc", segment.get("stopped_utc"))
        problems += _check_environment(f"{label}.environment", segment.get("environment"))
    return problems


def _check_block(label: str, block: object, keys: Sequence[str]) -> list[str]:
    """One optional sub-object whose keys are the contract: ``null``, or all of them present.

    Shared by the provenance views (``inputs.models``, ``inputs.data``) and, one level up, by the
    environment block: everywhere the record states a *set* of facts, an unset fact is an explicit
    ``null`` and a dropped key is a schema violation.
    """
    if block is None:
        return []
    if not isinstance(block, Mapping):
        return [f"{label}: must be an object or null"]
    return _check_keys(label, block, keys)


def _check_environment(label: str, environment: object) -> list[str]:
    """One environment block: ``null``, or an object carrying every key (story #139).

    Degradation is the *normal* case here — no git checkout, no ``psutil``, none of the optional
    extras — so the check is deliberately about **presence**, never about values: a bare core
    install must produce a schema-valid block whose facts are explicit nulls and whose
    ``degraded_seams`` names what was missing. A block that simply dropped the keys it could not
    answer would be indistinguishable from an older schema version, which is the one thing the
    record's explicit-null convention exists to prevent.
    """
    return _check_block(label, environment, _ENVIRONMENT_KEYS)


def _check_events(label: str, events: object) -> list[str]:
    if events is None:
        return []
    if not isinstance(events, Sequence) or isinstance(events, str | bytes):
        return [f"{label}: must be a list"]
    problems: list[str] = []
    for position, event in enumerate(events):
        if not isinstance(event, Mapping):
            problems.append(f"{label}[{position}]: must be an object")
            continue
        problems += _check_keys(f"{label}[{position}]", event, _EVENT_KEYS)
        problems += _check_stamp(f"{label}[{position}].t", event.get("t"))
    return problems


def _check_keys(label: str, section: Mapping[str, object], keys: Sequence[str]) -> list[str]:
    """Presence, not truthiness: an absent value is an explicit ``null``, never a missing key."""
    return [
        f"{label}.{key}: key is missing (absent values are explicit nulls)"
        for key in keys
        if key not in section
    ]


def _check_status(label: str, value: object, allowed: Sequence[str]) -> list[str]:
    if value in allowed:
        return []
    return [f"{label}: {value!r} is not one of {', '.join(allowed)}"]


def _check_stamp(label: str, value: object) -> list[str]:
    """UTC ISO-8601 with a ``Z`` marker, or ``null``. Anything else is ambiguous by timezone."""
    if value is None:
        return []
    if isinstance(value, str) and value.endswith("Z") and "T" in value:
        return []
    return [f"{label}: {value!r} is not a UTC ISO-8601 timestamp ending in 'Z'"]
