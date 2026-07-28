"""The run record's schema — its version, its caps, and a pure validator (story #129, epic #126).

One run, one self-describing file: ``workspace/runs/<run_id>/run.json``. This module holds the
contract that file promises to anything reading it (a website, a later Noctis, an operator with
``jq``), and nothing else — no I/O, no clock, no config, not even the record's construction. It is
deliberately the smallest module in the epic, because it is also the one the engine fingerprint
tracks as the ``schema`` component (``observability/engine_id.py``): changing what is recorded
should move a digest, and only that.

**Versioning is additive-only.** :data:`SCHEMA_VERSION` is 1. New fields may be added at any time;
an existing field never changes meaning or type, and a reader ignores keys it does not know (a
record carrying keys this Noctis has never heard of validates — story #143 pins that). That is what
lets a record written tonight still be read by the Noctis that resumes the run in a month. A
breaking change bumps the version and :func:`upgrade` rewrites the record **in place** on the next
open, recording what it did — a version is never silently repurposed and a key is never quietly
reinterpreted.

**Two conventions are part of the contract, not style.** Units are explicit in the field name
(``_usd``, ``_pct``, ``_bps``, ``_s``, ``_bytes``, ``_bars``, ``_hours``), and a known-absent value
is an explicit ``null`` rather than an omitted key — so a consumer can tell "not applicable" from
"this schema version did not have it". :func:`validate` enforces both structurally: every section's
keys must be *present* whatever their values, every dimensioned number must spell its unit the one
canonical way (:data:`UNIT_ALIASES`), and every timestamp key anywhere in the document — not only
in the sections with a hand-written rule — must be UTC ISO-8601 with a ``Z``.

**Caps are honest.** Bulky lists are bounded (:data:`TRADE_CAP`, :data:`EVENT_CAP`) and every cap
that bites writes a ``truncated`` note carrying kept/total counts. Silent truncation is forbidden:
a truncated record must never pass for a complete one. Segments and sessions are deliberately
uncapped — they are the run's spine, and losing one would make every derived total a lie.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "EVENT_CAP",
    "GATE_KEYS",
    "KIND",
    "PERFORMANCE_SOURCE",
    "PROMOTED_OUTCOME",
    "RECORD_SIZE_BUDGET_BYTES",
    "REJECTED_OUTCOME",
    "REQUIRED_SECTIONS",
    "RUN_STATUSES",
    "SCHEMA_VERSION",
    "SEGMENT_COMMANDS",
    "SEGMENT_STATUSES",
    "STAMP_KEYS",
    "STRATEGY_CAP",
    "STRATEGY_OUTCOMES",
    "STRATEGY_TIERS",
    "TRADE_CAP",
    "UNDECIDED_OUTCOME",
    "UNIT_ALIASES",
    "UPGRADES",
    "Upgrade",
    "upgrade",
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
    # Everything this run considered, with the structured gate evidence behind each verdict
    # (story #141). Present as a list — empty for a run that has considered nothing — never
    # ``null``: "no candidates yet" is a fact a young run can state, and the *rejections* are what
    # this section exists to publish.
    "strategies",
    # Every session this run's paper account closed, with its own trade log (story #142). A list —
    # empty for a run that never traded — never ``null``: "this run has closed no session" is a
    # fact, and the section is where the realised evidence behind ``performance`` is readable.
    "sessions",
    "performance",
    # The arena every number above was produced in (story #143). Mandatory and machine-readable:
    # a results page without its fill, holdout and promotion assumptions is not taken seriously,
    # and a page that states them in prose cannot be diffed between two runs. Never ``null`` —
    # the engine-fixed half (the fill model, the split geometry, the benchmark convention) is
    # true of every run, and the configured half is ``null`` key by key on a run that froze none.
    "assumptions",
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

# Bounds on the lists that grow without limit in a long run. A record is meant to be fetched
# whole by a website, so these keep a multi-week run in the low hundreds of kilobytes.
TRADE_CAP = 5_000
EVENT_CAP = 2_000
# Candidates carried in full. Deliberately generous: at the epic's own worked rate (66 candidates
# in a fortnight) this is roughly a year of nightly research, and the rejections are the product —
# dropping them cheaply would throw away exactly what the record exists to publish. When it does
# bite, champions are kept first (:func:`~noctis.reporting.run_record.build`) and the note beside
# it names the total.
STRATEGY_CAP = 500

# What a whole record is *meant* to weigh. Not enforced — a byte ceiling that truncated mid-write
# would be the silent truncation this contract forbids — but measured: a synthetic two-week run
# (14 segments, 66 candidates, 3 champions embedded in full, ~3 000 trials, 14 traded sessions at
# 30 fills each) is held under it by a test, so a change that quietly makes the record ten times
# heavier is a red test rather than a slow website.
#
# It states measured reality, and it has moved twice for that reason. The epic's planning estimate
# was ~40 KB; story #141's per-candidate gate evidence — the single largest thing in the strategies
# section, and the whole point of it — took the worked fortnight to ~172 KB, so the budget became
# 256 KiB. Story #142's realised record (the equity curve and the per-session trade log) adds ~110
# KB to the same fortnight, measured at 286 KB, so it is 384 KiB now.
#
# The caps above are a different instrument and should not be read as sized against this: they
# bound the *pathological* run (a record at :data:`TRADE_CAP` weighs well over a megabyte), while
# this describes the fortnight an operator actually gets. Both are honest — one is a ceiling that
# says so when it bites, the other is a measurement a test defends.
RECORD_SIZE_BUDGET_BYTES = 384 * 1024

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

# One considered candidate (story #141). Every candidate the run judged is here, not just the
# champions: "47 of 66 candidates died at the symbol-holdout gate" is the sentence that makes a
# results page credible, and it is computable only from the whole funnel.
_STRATEGY_KEYS = (
    "name",
    # promoted | rejected | undecided — the last verdict the champion board journaled for this
    # name, or ``undecided`` for a candidate that was authored and never carried to one.
    "outcome",
    # Which library tier the file sits in now (``champions`` / ``__tmp``), or ``null`` when the
    # run no longer has the file — a pruned run, or a draft that was swept.
    "tier",
    "decided_utc",
    # Trials journaled for this candidate, off its own ``state/experiments/<name>.jsonl``.
    "trials",
    # The structured evidence behind the verdict, in gate order.
    "gates",
    "rationale",
    # The path-plus-hash reference every non-champion carries instead of its text (see below).
    # The path is **relative to the run directory** the record sits in, so it stays portable
    # between machines and resolves beside the file quoting it.
    "source_path",
    "source_sha256",
    # The full text — for champions only, unless the run was created with ``embed_all_sources``.
    # ``null`` on every other candidate, which is what holds a fortnight's record to
    # :data:`RECORD_SIZE_BUDGET_BYTES` instead of a megabyte. The consequence is stated rather
    # than hidden: a rejected candidate's code is readable only while the run's tree survives
    # (``run.state_pruned`` says when it no longer does).
    "source",
)

# One gate's measurement (``champions/promotion.py``'s ``GateResult``). ``observed`` and
# ``threshold`` are ``null`` where a gate had nothing numeric to compare or the scorecard carried
# no such metric at all; ``note`` says why a gate could not bite.
#
# **The list is the gates *reached*, plus the one that failed.** A rejection short-circuits — the
# candidate was never measured against the gates after the one that stopped it — so a decision's
# evidence is a prefix of the gate order, and a consumer must read it as such: an absent gate means
# "not reached", never "passed". That is the honest shape, and it is why the funnel's death counts
# are read off the one entry whose ``passed`` is false.
GATE_KEYS = ("gate", "passed", "observed", "threshold", "note")

# A candidate's fate, as the record states it. ``undecided`` is a real and common state — a
# candidate authored, perhaps tuned, and never carried to a verdict — and it is evidence of the
# funnel's width above the gates, not a hole in the record.
PROMOTED_OUTCOME = "promoted"
REJECTED_OUTCOME = "rejected"
UNDECIDED_OUTCOME = "undecided"
STRATEGY_OUTCOMES = (PROMOTED_OUTCOME, REJECTED_OUTCOME, UNDECIDED_OUTCOME)

# The two writable library tiers a run owns (the committed ``strategies/`` seeds are read-only
# input and belong to no run, so they are never a candidate's tier).
STRATEGY_TIERS = ("champions", "__tmp")

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

# One closed trading session (story #142) — the realised evidence the performance block is derived
# from. ``equity`` is the *account's* mark at that close (the curve); ``start_equity`` /
# ``end_equity`` are the session's own bounds, which differ whenever positions were carried in.
_SESSION_KEYS = (
    "as_of",
    "equity",
    "start_equity",
    "end_equity",
    "realized_pnl",
    "orders_submitted",
    "positions_end",
    "trades",
)

# One fill, with the four fields story #142 added: when it happened, what it cost in fees and
# modelled slippage, and **which champion** the symbol was assigned when it filled. The last is
# what makes the trade log readable per champion instead of as one blended blob.
_TRADE_KEYS = (
    "ts",
    "symbol",
    "side",
    "quantity",
    "price",
    "fees_usd",
    "slippage_bps",
    "champion",
    "rationale",
)

# The realised paper-account record (story #142). Present only for a run that traded — the
# ``traded: false`` ⇒ ``null`` pairing below is the contract — and then carrying every section,
# with ``null`` wherever a metric's inputs do not exist yet.
_PERFORMANCE_KEYS = (
    # The section names what produced it. Backtest and scorecard numbers live under
    # ``strategies[].scorecard`` and are never blended in; a consumer that renders both can tell
    # them apart structurally rather than by convention.
    "source",
    "account",
    "equity_curve",
    "returns",
    "risk_adjusted",
    "drawdown",
    "trades",
    "benchmark",
    "monthly_returns_pct",
)
_RISK_ADJUSTED_KEYS = (
    "sharpe",
    "sortino",
    "calmar",
    # The Probabilistic Sharpe Ratio, and the Deflated one beside the trial count that deflated it
    # — ``n_trials_used`` is part of the contract, because a deflation nobody can audit is a
    # number nobody should trust. ``deflation_basis`` names the variance assumption behind it.
    "psr",
    "deflated_sharpe",
    "n_trials_used",
    "deflation_basis",
    "skew",
    "excess_kurtosis",
    "annualization_basis",
)
_DRAWDOWN_KEYS = (
    "max_drawdown_pct",
    "max_drawdown_days",
    "peak_date",
    "trough_date",
    "recovered",
    "recovery_factor",
)
_BENCHMARK_KEYS = (
    # Named ``equal_weight_universe_bh``, never "the index": it is buy-and-hold over the names this
    # run traded, priced from bars already in the lake. ``note`` carries why a number is missing —
    # a run whose symbols the lake does not hold is not benchmarked rather than benchmarked wrongly.
    "name",
    "method",
    "symbols",
    "total_return_pct",
    "sharpe",
    "alpha_pct",
    "beta",
    "information_ratio",
    "tracking_error_pct",
    "correlation",
    "note",
)

# The one name the realised block may call itself, checked rather than assumed: a section that
# renamed its source could be presented beside a scorecard as if the two were the same measurement.
PERFORMANCE_SOURCE = "paper_account"

# The marker every dollar figure in the record must carry. Prices come from a versioned list-price
# table and are therefore *estimates*; a field that named itself ``usd`` alone would read as a
# receipt, and the difference matters most to exactly the reader who is comparing two runs on cost.
USD_MARKER = "usd"
ESTIMATE_MARKER = "estimate"

# The honesty block (story #143) — the arena, as data. Every key present on every record: the
# engine-fixed half is true of any run, and the configured half is an explicit ``null`` on a run
# that froze no configuration, because "this run never recorded its arena" and "this run charged
# no fees" are very different statements about a result.
_ASSUMPTIONS_KEYS = (
    # Measured off the safety gate's own resolved verdict, never asserted (AGENTS.md rule 1).
    "paper_only",
    "live_gate",
    "fill_model",
    "lookahead",
    "fee_bps",
    "slippage_bps",
    "round_trip_cost_bps",
    "costs_charged",
    "walk_forward",
    "forward_holdout",
    "symbol_holdout",
    # The exhaustion gate's floor: the distinct param sets a verdict may not be reached without.
    "min_trials",
    # The whole promotion subtree, verbatim — see ``reporting/assumptions.py`` for why the block
    # publishes the settings rather than a curated list of them.
    "promotion_thresholds",
    "benchmark",
    "state_scope",
)
_LIVE_GATE_KEYS = (
    # The gate's verdict for this run. The two settings behind it (``mode``, ``allow_live``) are
    # never recorded — see :data:`UNRECORDABLE_SETTINGS`.
    "execution_mode",
    "real_orders_reachable",
    "re_resolved_each_segment",
    "note",
)
_WALK_FORWARD_KEYS = (
    "sizing",
    "min_train_bars",
    "max_train_bars",
    "min_test_bars",
    "max_test_bars",
    # ``null`` means one test window: consecutive splits advance by exactly the test size.
    "step_bars",
    "embargo_bars",
    "test_after_train",
)
_FORWARD_HOLDOUT_KEYS = ("min_bars", "max_bars", "reserved", "note")
_SYMBOL_HOLDOUT_KEYS = ("size", "fit_set_size", "symbols")
_BENCHMARK_ASSUMPTION_KEYS = ("name", "method", "rebalancing")

# The two settings a record may never carry, whatever else it grows: the live-money double gate.
# The safety gate re-resolves from the config file and the ALLOW_LIVE environment variable at every
# process start, so a record that carried either one could offer a second source for a decision
# that must have exactly two independent ones (AGENTS.md rule 1).
UNRECORDABLE_SETTINGS = ("mode", "allow_live")

# The contract's unit vocabulary, as a **refusal table**: the long or ambiguous spelling of a
# dimension, mapped to the one suffix the record uses for it. A number whose key ends in a key of
# this table is a schema violation, wherever in the document it sits — which is what makes "units
# are explicit and canonical" a check rather than a convention. The rule deliberately bites on
# *numbers only*: a section named for what its values measure (``phase_seconds``, a mapping of
# phase → seconds) is a name, while ``cumulative_runtime_seconds`` would be a number lying about
# which spelling this record uses, and only one of those confuses a consumer indexing by key.
UNIT_ALIASES: Mapping[str, str] = {
    "seconds": "_s",
    "secs": "_s",
    "sec": "_s",
    "millis": "_s",
    "milliseconds": "_s",
    "ms": "_s",
    "percent": "_pct",
    "percentage": "_pct",
    "dollars": "_usd",
    "dollar": "_usd",
    "cents": "_usd",
    "bp": "_bps",
    "basispoints": "_bps",
    "points": "_bps",
    "kb": "_bytes",
    "mb": "_bytes",
    "gb": "_bytes",
    "kib": "_bytes",
    "mib": "_bytes",
}

# The subtrees the unit rule does **not** reach, because they are not the record's vocabulary:
# they are foreign documents quoted verbatim. ``inputs.settings.resolved`` is the operator's own
# configuration as the run froze it (``research_time_budget_minutes`` is a config key, not a
# record field), and ``inputs.mandate`` carries the mandate's front-matter overlay as written.
# Renaming a key inside either would make the record disagree with the file it claims to quote —
# a worse failure than a second spelling of a unit, and one nobody could debug.
_VERBATIM_SUBTREES = ("inputs.settings.resolved", "inputs.mandate")

# Every key in the record that carries a moment. Checked **structurally**, over the whole
# document, so a section added later inherits the rule instead of needing its own line here: a
# stamp that lost its ``Z`` is ambiguous by timezone, and a website plotting it would be wrong by
# hours without anything looking broken.
STAMP_KEYS = ("t", "at", "ts")
_STAMP_SUFFIX = "_utc"


@dataclass(frozen=True)
class Upgrade:
    """What reading an older record under this engine did to it (:func:`upgrade`).

    ``record`` is a **new** document — nothing here mutates its input — carrying the target
    version and whatever the registered steps rewrote. ``applied`` names each step that ran, and
    is empty for the ordinary additive bump, where the record is simply restamped: an
    additive-only change adds keys, and the record is rebuilt from the run's own artifacts at
    every write anyway, so the new keys arrive with the next flush rather than through a
    translation. A step exists for the one case additive-only does not cover — a value that has
    to be *carried* into a new shape — and the registry is where such a case is reviewed.
    """

    record: dict
    from_version: int | None
    to_version: int
    applied: tuple[str, ...] = ()

    @property
    def upgraded(self) -> bool:
        """Whether this reading moved the record's version at all."""
        return self.from_version != self.to_version

    def note(self) -> str | None:
        """The one sentence the run record files as an event, or ``None`` when nothing moved.

        A schema upgrade is exactly the kind of thing that must never happen quietly: a consumer
        comparing two of this run's segments has to be able to see that the document changed shape
        under them, and the run's own event stream is where the run says what happened to it.
        """
        if not self.upgraded:
            return None
        origin = "no version" if self.from_version is None else f"version {self.from_version}"
        detail = f" ({'; '.join(self.applied)})" if self.applied else ""
        return (
            f"this run's record was written under schema {origin} and was upgraded in place to "
            f"version {self.to_version} by this engine{detail}"
        )


# The registered translations, keyed by the version each one upgrades **from**. Empty at
# :data:`SCHEMA_VERSION` 1 — there is no earlier version to come from — and deliberately kept as a
# registry rather than an ``if`` ladder, so a version that needs to carry a value into a new shape
# is one reviewable entry and the walk over intermediate versions never has to change.
UPGRADES: Mapping[int, Callable[[Mapping[str, object]], dict]] = {}


def upgrade(
    record: Mapping[str, object],
    *,
    upgrades: Mapping[int, Callable[[Mapping[str, object]], dict]] | None = None,
    target: int | None = None,
) -> Upgrade:
    """Bring one record up to this engine's schema version. Pure: a new document is returned.

    The promise this implements is the one the epic makes to a multi-week run: **today's run must
    still be resumable by tomorrow's engine**. A record below :data:`SCHEMA_VERSION` is walked up
    one version at a time — each registered step in :data:`UPGRADES` applied in order, so a record
    two versions behind is not special-cased — and restamped. The result is written back by the
    ordinary record write, which is what makes the upgrade happen *in place*, and the returned
    :meth:`Upgrade.note` is what puts it on the run's own event stream.

    A record from the **future** is never touched: additive-only means a newer document is
    readable by ignoring what this engine does not know, and rewriting its version down would
    destroy exactly the information a later reader needs. A record with no version at all is
    treated as predating versioning and upgraded from nothing.

    Both seams are injectable so the mechanism can be exercised against a synthetic later version
    without bumping the real one — a versioning path first tested on the day it is first needed is
    a versioning path nobody has tested.
    """
    steps = UPGRADES if upgrades is None else upgrades
    to_version = SCHEMA_VERSION if target is None else target
    found = record.get("schema_version")
    from_version = found if isinstance(found, int) and not isinstance(found, bool) else None
    if from_version is not None and from_version >= to_version:
        return Upgrade(record=dict(record), from_version=from_version, to_version=from_version)
    document = dict(record)
    applied: list[str] = []
    version = from_version
    while version is not None and version < to_version:
        step = steps.get(version)
        if step is None:
            break
        document = dict(step(document))
        applied.append(f"{version}→{version + 1}")
        version += 1
    document["schema_version"] = to_version
    return Upgrade(
        record=document,
        from_version=from_version,
        to_version=to_version,
        applied=tuple(applied),
    )


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
    problems += _check_strategies(record.get("strategies"))
    problems += _check_spend(record.get("spend"))
    problems += _check_sessions(record.get("sessions"))
    problems += _check_performance(record.get("performance"), run=run)
    problems += _check_assumptions(record.get("assumptions"))

    for name in ("events", "errors"):
        problems += _check_events(name, record.get(name))
    # The two contract-wide conventions, checked over the whole document rather than section by
    # section — so a section added tomorrow inherits both rules instead of needing its own line.
    problems += _check_units("", record)
    problems += _check_stamps("", record)
    return problems


def _check_assumptions(assumptions: object) -> list[str]:
    """The honesty block: present, complete, and never a second source for the live-money gates.

    Presence is the whole check on the *values*, deliberately. What the arena was is a fact about
    the run, not something this module can second-guess — but a **missing** key is different in
    kind: a website rendering the honesty table would silently drop the row, and a reader diffing
    two runs would see "not stated" where the schema promises "stated, possibly null".
    """
    if assumptions is None:
        return ["assumptions: section must be an object — an arena is stated on every record"]
    if not isinstance(assumptions, Mapping):
        return ["assumptions: section must be an object"]
    problems = _check_keys("assumptions", assumptions, _ASSUMPTIONS_KEYS)
    problems += _check_block("assumptions.live_gate", assumptions.get("live_gate"), _LIVE_GATE_KEYS)
    problems += _check_block(
        "assumptions.walk_forward", assumptions.get("walk_forward"), _WALK_FORWARD_KEYS
    )
    problems += _check_block(
        "assumptions.forward_holdout", assumptions.get("forward_holdout"), _FORWARD_HOLDOUT_KEYS
    )
    problems += _check_block(
        "assumptions.symbol_holdout", assumptions.get("symbol_holdout"), _SYMBOL_HOLDOUT_KEYS
    )
    problems += _check_block(
        "assumptions.benchmark", assumptions.get("benchmark"), _BENCHMARK_ASSUMPTION_KEYS
    )
    gate = assumptions.get("live_gate")
    section: Mapping[str, object] = gate if isinstance(gate, Mapping) else {}
    return problems + [
        f"{label}.{key}: the live-money gates are never recorded — this block carries the gate's "
        "verdict, and the pair behind it has exactly two independent sources, neither of them a "
        "record"
        for label, block in (("assumptions", assumptions), ("assumptions.live_gate", section))
        for key in UNRECORDABLE_SETTINGS
        if key in block
    ]


def _check_units(label: str, node: object) -> list[str]:
    """Every dimensioned number in the document spells its unit the one canonical way.

    Walks the whole record except the subtrees it *quotes* (:data:`_VERBATIM_SUBTREES`). A
    *number* whose key ends in a non-canonical spelling of a unit (:data:`UNIT_ALIASES`) is a
    violation: a consumer indexing by key — a website's honesty table, a diff between two runs, a
    ``jq`` filter — has to be able to rely on ``_s`` meaning seconds everywhere, and one field
    spelled ``_seconds`` is exactly the drift that makes it stop.
    """
    if label in _VERBATIM_SUBTREES:
        return []
    problems: list[str] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            name = str(key)
            path = f"{label}.{name}" if label else name
            if isinstance(value, int | float) and not isinstance(value, bool):
                canonical = UNIT_ALIASES.get(name.rsplit("_", 1)[-1].lower())
                if canonical is not None:
                    problems.append(
                        f"{path}: a dimensioned number names its unit canonically — use the "
                        f"{canonical!r} suffix, not {name.rsplit('_', 1)[-1]!r}"
                    )
            problems += _check_units(path, value)
    elif isinstance(node, Sequence) and not isinstance(node, str | bytes):
        for position, item in enumerate(node):
            problems += _check_units(f"{label}[{position}]", item)
    return problems


def _check_stamps(label: str, node: object) -> list[str]:
    """Every timestamp anywhere in the document is UTC ISO-8601 with a ``Z``.

    The structural twin of the per-section stamp checks above, and the reason a new section needs
    no new rule: a key named ``…_utc`` (or one of :data:`STAMP_KEYS`) carrying a string must carry
    a stamp a reader can place on a timeline without guessing a timezone.
    """
    problems: list[str] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            name = str(key)
            path = f"{label}.{name}" if label else name
            if isinstance(value, str) and (name.endswith(_STAMP_SUFFIX) or name in STAMP_KEYS):
                problems += _check_stamp(path, value)
            problems += _check_stamps(path, value)
    elif isinstance(node, Sequence) and not isinstance(node, str | bytes):
        for position, item in enumerate(node):
            problems += _check_stamps(f"{label}[{position}]", item)
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
    problems = _check_keys("performance", performance, _PERFORMANCE_KEYS)
    if performance.get("source") != PERFORMANCE_SOURCE:
        problems.append(
            f"performance.source: expected {PERFORMANCE_SOURCE!r} — this section is the paper "
            "account's realised record, and a backtest scorecard must never be presented as one"
        )
    problems += _check_block(
        "performance.risk_adjusted", performance.get("risk_adjusted"), _RISK_ADJUSTED_KEYS
    )
    problems += _check_block("performance.drawdown", performance.get("drawdown"), _DRAWDOWN_KEYS)
    problems += _check_block("performance.benchmark", performance.get("benchmark"), _BENCHMARK_KEYS)
    risk = performance.get("risk_adjusted")
    if isinstance(risk, Mapping) and risk.get("deflated_sharpe") is not None:
        if risk.get("n_trials_used") is None:
            problems.append(
                "performance.risk_adjusted.n_trials_used: a deflated Sharpe must publish the "
                "trial count it was deflated by, or the deflation cannot be audited"
            )
    return problems


def _check_sessions(sessions: object) -> list[str]:
    """One entry per closed session, each carrying its own trade log.

    Uncapped, like the segments: the sessions are the run's realised spine and losing one would
    make every derived total a lie. The *trades* inside them are bounded (``TRADE_CAP``), and a
    bounded log writes its note in ``run.truncated``.
    """
    if sessions is None:
        return []
    if not isinstance(sessions, Sequence) or isinstance(sessions, str | bytes):
        return ["sessions: must be a list"]
    problems: list[str] = []
    for position, session in enumerate(sessions):
        label = f"sessions[{position}]"
        if not isinstance(session, Mapping):
            problems.append(f"{label}: must be an object")
            continue
        problems += _check_keys(label, session, _SESSION_KEYS)
        trades = session.get("trades")
        if trades is None or isinstance(trades, str | bytes) or not isinstance(trades, Sequence):
            problems.append(f"{label}.trades: must be a list")
            continue
        for index, trade in enumerate(trades):
            entry = f"{label}.trades[{index}]"
            if not isinstance(trade, Mapping):
                problems.append(f"{entry}: must be an object")
                continue
            problems += _check_keys(entry, trade, _TRADE_KEYS)
    return problems


def _check_strategies(strategies: object) -> list[str]:
    """Everything the run considered: one entry per candidate, each carrying its gate evidence.

    Two rules beyond presence, both of which exist because a consumer would otherwise be misled
    rather than merely under-informed:

    * a **gate entry** carries all five keys, so "which gate stopped this candidate, and by how
      much" is answerable without guessing what a missing key meant;
    * an **embedded source** carries the hash of what was embedded. A record that quoted code with
      no digest beside it could not be checked against the file it claims to be, and the reference
      form (path + hash) is the whole contract for every candidate that is *not* embedded.
    """
    if strategies is None:
        return []
    if not isinstance(strategies, Sequence) or isinstance(strategies, str | bytes):
        return ["strategies: must be a list"]
    problems: list[str] = []
    for position, strategy in enumerate(strategies):
        label = f"strategies[{position}]"
        if not isinstance(strategy, Mapping):
            problems.append(f"{label}: must be an object")
            continue
        problems += _check_keys(label, strategy, _STRATEGY_KEYS)
        problems += _check_status(f"{label}.outcome", strategy.get("outcome"), STRATEGY_OUTCOMES)
        tier = strategy.get("tier")
        if tier is not None:
            problems += _check_status(f"{label}.tier", tier, STRATEGY_TIERS)
        if strategy.get("source") is not None and strategy.get("source_sha256") is None:
            problems.append(
                f"{label}.source_sha256: an embedded source must carry the hash of what was "
                "embedded, or nothing can check the record against the file it quotes"
            )
        problems += _check_gates(f"{label}.gates", strategy.get("gates"))
    return problems


def _check_gates(label: str, gates: object) -> list[str]:
    """One decision's gate evidence — the gates reached, plus the one that failed."""
    if gates is None:
        return []
    if not isinstance(gates, Sequence) or isinstance(gates, str | bytes):
        return [f"{label}: must be a list"]
    problems: list[str] = []
    for position, gate in enumerate(gates):
        entry = f"{label}[{position}]"
        if not isinstance(gate, Mapping):
            problems.append(f"{entry}: must be an object")
            continue
        problems += _check_keys(entry, gate, GATE_KEYS)
    return problems


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
