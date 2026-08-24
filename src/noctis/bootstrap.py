"""The composition root — every entrypoint assembles its session here.

Before this module, session assembly was scattered: the ``promotion.metric`` precedence
chain (``config.yaml`` → mandate overlay → ``--metric`` flag) spanned four files with its
ordering enforced by comments, :class:`~noctis.champions.promotion.PromotionRules` was
hand-built from settings in two places, and the CLI and the runtime each wired their own
copy of the agent research session (client + budgets + toolbox + loop kwargs).

Everything here is plain assembly, no policy: the safety gate, the settings-overlay
classifier, and the budget tables all stay with their owners (``config.gate``,
``config.overlay``, ``research.cost``). This module only fixes the *order* in one place,
wraps every overlay it performs in the gate-unmoved assertion (:func:`overlay_mandate`),
and hands back built
collaborators. Errors are typed, never printed — the CLI maps them to red text + exit
codes; a library caller sees ordinary exceptions.

Heavy collaborators import at call time, mirroring the CLI convention (fast ``--help``)
and keeping test monkeypatching on the owning modules effective.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from noctis.config import Settings, load_settings, resolve_execution_mode
from noctis.config.settings import SECRET_FIELDS

# The sink contract, not a collaborator: NULL_SINK is the default value every session object
# below declares, so it imports at module load like the settings types do. Core-only — the
# events module names no renderer and no heavy package.
from noctis.observability.events import NULL_SINK, EventSink

logger = logging.getLogger("noctis.bootstrap")

if TYPE_CHECKING:
    from noctis.champions.promotion import PromotionRules
    from noctis.data.seam import MarketDataLake
    from noctis.engine.research import ResearchSummary
    from noctis.observability import Event
    from noctis.observability.debug import Recorder
    from noctis.reporting.run_tree import RunStore
    from noctis.research import CostProfile, Mandate, ResearchToolbox
    from noctis.strategies.families import FamilyRegistry


class MissingVendorKey(RuntimeError):
    """A command that must fetch data was started without a vendor credential."""


class UsageError(ValueError):
    """Mutually-exclusive or unknown session flags. Distinct from :class:`ValueError` so a
    CLI handler never mistakes a pydantic ``ValidationError`` (also a ValueError) for usage."""


class RunPrunedError(RuntimeError):
    """A reading of a run whose heavy directories retention already deleted (epic #292).

    Raised by :func:`open_reading` when the addressed record says ``state_pruned`` and the verb did
    not say it can read one anyway (``readable_pruned``). Prunability is decided **once**, here, so
    the two policies stay one keyword apart: a verb that would assemble something out of the run's
    ``state/`` or ``reports/`` (``report``, ``champions``, ``account``, ``backtest``,
    ``strategies``, ``status``) is refused rather than shown a board that reads as empty for a run
    that had one, while the verbs whose whole subject is the record (``run-record``, ``--finish``,
    ``run-prune``) still reach it — the record *is* the history retention kept.
    """


@dataclass(frozen=True)
class OverrideChange:
    """One setting a mandate's overlay moved, from what to what.

    ``before`` is the value **config** resolved for the path (``config.yaml`` + ``.env`` + the
    environment) and ``after`` is the value the session will actually use — the same value the
    ``"k=v"`` echo line carries. Read off the live settings object either side of the one
    guarded overlay seam (:func:`~noctis.config.overlay.patch_snapshot`), never re-derived, so
    a preflight can render the change without owning a second copy of the precedence chain.
    """

    path: str
    before: Any
    after: Any


@dataclass(frozen=True)
class SessionInputs:
    """The resolved inputs of one session: settings after every override, plus provenance."""

    settings: Settings
    # The gate-resolved execution mode, or None when the entrypoint didn't ask for the gate.
    # Every verb that **opens a run** arms it (``run`` and — since story #247 — ``research``): the
    # run freezes the verdict at creation and may trade on a later segment, so a record must carry
    # what the gate decided rather than "nobody measured".
    # A verb that only *reads* resolves no mode, and that is the shape :func:`open_reading` hands
    # back on a :class:`Reading` (``require_gate=False`` by default, epic #292): a reading places no
    # order, so eleven readers gain no refusal for no safety gain. ``status`` is the exception at
    # both ends — it arms the gate because it **narrates** the resolved mode, and a mode line it had
    # to guess at would be worse than none. The preflights (``mandate``, ``engine``, the
    # config-drift preview) resolve none either.
    mode: Literal["paper", "live"] | None
    mandate: Mandate | None
    # "k=v" echo lines for each mandate override actually applied (the CLI prints them).
    overrides: list[str]
    # The same overrides with their pre-overlay value alongside, path-sorted — what `noctis
    # mandate` renders as the effective settings diff. Same paths as ``overrides`` by
    # construction (both come from the one applied patch); empty when nothing was overlaid.
    changes: list[OverrideChange] = field(default_factory=list)
    # The run address this session continues — the one an operator *typed* (an id, ``latest``, a
    # path, ``@label``), not the id it resolved to, so every refusal, echo and record write
    # downstream is about a run they can go and look at. ``None`` on a fresh run: a run being
    # minted right now continues nothing. Whether this session resumes at all is exactly
    # ``resume is not None`` — one field, so no entrypoint can carry the address and the flag apart.
    resume: str | None = None
    # The run's frozen ``inputs`` re-frozen on the current files, when this session is a
    # ``--rebase-config`` resume that found something to adopt (story #134): epoch already bumped,
    # before/after entry already appended, built once by ``config.rehydrate.rebase_inputs``. Read
    # off this session by :func:`open_run_store`, which lets it replace what the record carried.
    # ``None`` on every other session — including a rebase of a run nothing changed under, which is
    # a no-op by design rather than a cosmetic epoch bump.
    rebase: Mapping[str, Any] | None = None
    # What the engine-change resume policy found (story #135), for the entrypoint to *say and
    # record* once the run is open: one event text per tier that moved. Empty on a fresh run and on
    # a resume that found no drift — silence is the signal that the engine held still.
    engine_notes: list[str] = field(default_factory=list)
    # The ``engine_changes`` entry a deliberately accepted engine change produced
    # (``--allow-engine-upgrade``): epoch already bumped, moved components already named. Read off
    # this session by :func:`open_run_store`, which re-freezes the run onto this engine — like every
    # other field freezing reads, so no entrypoint names it. ``None`` everywhere else,
    # including an upgrade of a run whose arbiter never moved — a documented no-op.
    engine_upgrade: Mapping[str, Any] | None = None


def resolve_session(
    config_path: str | None = None,
    *,
    directive: str | None = None,
    mandate: str | None = None,
    metric: str | None = None,
    time_limit_hours: float | None = None,
    run_limit_hours: float | None = None,
    embed_all_sources: bool = False,
    require_gate: bool = False,
    resume: str | None = None,
    rebase_config: bool = False,
    allow_engine_upgrade: bool = False,
) -> SessionInputs:
    """Resolve one session's settings by the one precedence order (docs/configuration.md).

    ``load_settings`` → safety gate (when ``require_gate``) → ``resolve_mandate`` →
    :func:`overlay_mandate` → explicit CLI flags last, so a one-off ``--metric`` still wins
    over a mandate's overlay. The gate resolves *before* the overlay because nothing
    downstream may run against an un-gated mode; the gate-unmoved assertion then proves the
    overlay never reached it anyway. Raises :class:`UsageError` on bad flags (both mandate
    selectors, an unknown metric), :class:`~noctis.research.MandateError` on an unresolvable
    selector, :class:`~noctis.config.SafetyGateError` when the gate refuses, and
    :class:`~noctis.config.OverlayError` if an overlay moved a refused setting — all before
    any long-running work starts.

    ``resume`` continues an existing run (:func:`resume_session`): the middle of the chain —
    reading ``mandate/``, applying its overlay — is replaced by the run's **frozen** config, while
    the ends are untouched. ``load_settings`` still runs (the live tier: paths, secrets,
    per-process budgets), the safety gate still resolves first and fresh, and the explicit CLI
    flags still land last. ``rebase_config`` (story #134) puts that middle back for one session:
    the current files are read, resolved and **adopted** onto the run.
    """
    from noctis.backtest.scorecard import Metric

    if directive is not None and mandate is not None:
        raise UsageError("Pass either --directive or --mandate, not both.")
    if metric is not None:
        try:
            Metric.parse(metric)
        except ValueError as exc:  # the one diagnosis, re-typed as a usage error
            raise UsageError(str(exc)) from None

    if resume is not None:
        if directive is not None or mandate is not None:
            raise UsageError(
                "A resumed run's mandate is frozen at creation, so --directive/--mandate cannot "
                "steer it: its accumulated results mean what the original mandate asked for. "
                "Start a new run to research something else."
            )
        if metric is not None:
            raise UsageError(
                "A resumed run's election metric is frozen at creation, so --metric cannot move "
                "it: champions crowned under two metrics were never comparable, and the metric "
                "is part of the run's comparability key. Start a new run to score differently."
            )
        if embed_all_sources:
            raise UsageError(
                "A run's source-embedding choice is frozen at creation, so --embed-all-sources "
                "cannot move it: the record is rewritten whole at every write, and a flag that "
                "could be passed on some nights and not others would make what the record "
                "contains depend on how it was last invoked. Every candidate is already named in "
                "the record by path and content hash — read them from the run's own "
                "strategies/__tmp/ tier, which survives as long as the run does."
            )
        if run_limit_hours is not None:
            raise UsageError(
                "A run's compute cap is frozen at creation, so --run-limit-hours cannot move it: "
                "the cap is what makes two runs comparable on equal compute, and a cap that could "
                "be raised each night would bound nothing at all. Use --time-limit-hours to bound "
                "tonight, or start a new run to give a fresh experiment a different budget."
            )
        return resume_session(
            config_path,
            run_id=resume,
            time_limit_hours=time_limit_hours,
            require_gate=require_gate,
            rebase_config=rebase_config,
            allow_engine_upgrade=allow_engine_upgrade,
        )
    if rebase_config:
        raise UsageError(
            "--rebase-config adopts the current config.yaml and mandate/ onto an existing run, so "
            "it only means something with --resume: a run being minted right now is already being "
            "frozen on exactly those files."
        )
    if allow_engine_upgrade:
        raise UsageError(
            "--allow-engine-upgrade accepts an engine change on an existing run, so it only means "
            "something with --resume: a run being minted right now is being frozen on exactly the "
            "engine this process is."
        )

    resolved = _session_from_files(
        config_path, require_gate=require_gate, directive=directive, mandate=mandate
    )
    settings = resolved.settings
    if metric is not None:
        settings.promotion.metric = metric
    if time_limit_hours is not None:
        settings.time_limit_hours = time_limit_hours
    if run_limit_hours is not None:
        # Frozen tier, and this is the one moment it may be set: a run is being minted right now,
        # so the flag is part of what this experiment *is*. Every later segment restores it.
        settings.run_limit_hours = run_limit_hours
    if embed_all_sources:
        # The same rule for the same reason (story #141): what the run's record archives is part
        # of what the run is, decided once, here.
        settings.embed_all_sources = True
    return resolved


def _session_from_files(
    config_path: str | None,
    *,
    require_gate: bool,
    directive: str | None = None,
    mandate: str | None = None,
    mandate_overlay: bool = True,
) -> SessionInputs:
    """The precedence chain over the **current files**: settings → gate (when asked) → overlay.

    Everything :func:`resolve_session` does except the explicit CLI flags, which land after it and
    are the one thing a *reading* never applies — so a session being minted and a reading of the
    reserved run are told the same thing by construction (:func:`open_reading` is the other
    caller). No path is bound and no run is addressed here: the reserved ``legacy`` tree is what
    settings assembled from the current files already point at.

    ``mandate_overlay=False`` skips ``resolve_mandate``/:func:`overlay_mandate` entirely and yields
    ``mandate=None, overrides=[]``. That is ``status``'s degrade, not a second policy: a diagnosis
    verb whose mandate is the problem must still print the mode line it exists to print.
    """
    from noctis.config.overlay import patch_snapshot
    from noctis.research import resolve_mandate

    settings = load_settings(config_path=config_path)
    mode = resolve_execution_mode(settings) if require_gate else None
    if not mandate_overlay:
        return SessionInputs(settings=settings, mode=mode, mandate=None, overrides=[])
    active = resolve_mandate(settings, cli_directive=directive, cli_mandate=mandate)
    # The overlay is applied once, and both readings of it are taken around that one call: the
    # applier's ``"k=v"`` echo lines, and the same paths' values snapshotted either side. The
    # pre-values have to be captured *here* — after config resolved and before the patch lands
    # — because nothing downstream can recover them; ``noctis mandate`` renders the pair as the
    # effective settings diff without owning a second copy of the precedence chain.
    patch = active.config_overrides if active is not None else {}
    before = patch_snapshot(settings, patch)
    overrides = overlay_mandate(settings, active)
    changes = [
        OverrideChange(path=path, before=before[path], after=after)
        for path, after in sorted(patch_snapshot(settings, patch).items())
    ]
    warn_if_auto_overlay_is_inert(settings, active)
    return SessionInputs(
        settings=settings, mode=mode, mandate=active, overrides=overrides, changes=changes
    )


def resume_session(
    config_path: str | None = None,
    *,
    run_id: str,
    time_limit_hours: float | None = None,
    require_gate: bool = False,
    rebase_config: bool = False,
    allow_engine_upgrade: bool = False,
) -> SessionInputs:
    """Resolve the session that **continues** an existing run, under that run's frozen config.

    The whole point of the run record (epic #126): a run is stopped each morning and resumed each
    night, and its numbers only mean something if the configuration that produced them held still
    in between. So the current ``config.yaml`` and ``mandate/`` are read for the *live* tier only —
    paths, secrets, per-process budgets — and everything that decides what the results mean comes
    back from the record (:mod:`noctis.config.rehydrate`).

    Four refusals, all before any long-running work starts and all from somewhere else: the
    address must name a run (:class:`~noctis.reporting.run_tree.RunNotFoundError`), the run must
    not be ``completed`` (:class:`~noctis.reporting.run_tree.RunCompletedError`), the freshly
    resolved execution mode must match the one the run's earlier segments ran under
    (:class:`~noctis.config.rehydrate.RehydrationError`), and the **arbiter** of the engine — what
    passes, and what a number means — must still be the one the run was created under
    (:class:`~noctis.observability.engine_change.EngineChangeError`, story #135). The safety gate
    itself is re-resolved here exactly as at a first start — never rehydrated, never restored
    (AGENTS.md rule 1).

    An engine change in the *searcher* tier is not a refusal at all: it is warned about, handed
    back for the entrypoint to record against the run, and the resume proceeds.
    ``allow_engine_upgrade`` is the operator accepting an arbiter change deliberately, which lifts
    that one refusal and produces the record entry that makes the acceptance permanent.

    Drift between the record and the current files is normal and silently fine: frozen wins.
    ``rebase_config`` (story #134) is how an operator adopts it instead — deliberately, once, and
    on the record: the current ``config.yaml`` and ``mandate/`` are resolved exactly as on a first
    start and re-frozen onto the run with the epoch bumped and a before/after entry appended
    (:func:`~noctis.config.rehydrate.rebase_inputs`). With nothing to adopt it is a **no-op**: this
    falls through to the ordinary resume rather than bumping an epoch for a change that never
    happened. It never reaches the refused tier — ``mode``/``allow_live`` are checked before it and
    refused with a message that says no flag lifts them.
    """
    from noctis.config.rehydrate import assert_mode_unchanged
    from noctis.reporting.run_tree import assert_resumable, read_run_record

    settings = load_settings(config_path=config_path)
    mode = resolve_execution_mode(settings) if require_gate else None
    record = read_run_record(settings.runs_dir, run_id)
    # ``run_id`` is the *address* an operator typed (an id, `latest`, a path, `@label`); the run's
    # own id is what the record it resolved to says. Every refusal and warning below names that,
    # so a refusal is always about a run an operator can go and look at.
    addressed = _addressed_id(record, run_id)
    assert_resumable(record, addressed)
    if mode is not None:
        assert_mode_unchanged(record, mode, rebasing=rebase_config)
    notes, upgrade = _engine_change_on_resume(
        record, run_id=addressed, upgrading=allow_engine_upgrade
    )
    if rebase_config:
        adopted = _adopt_current_config(
            settings, record, mode=mode, time_limit_hours=time_limit_hours, run_id=addressed
        )
        if adopted is not None:
            # The engine verdict rides along whichever way the config went: a rebase adopts new
            # *settings*, and says nothing about the code that will run them. A rebase is still a
            # resume, so it carries the same address every other resumed session does.
            return replace(adopted, resume=run_id, engine_notes=notes, engine_upgrade=upgrade)
    frozen = _session_from_record(record, settings, mode=mode, run_id=addressed)
    # The live tier's last word, exactly as on a first start: an explicit flag beats the file it
    # would have come from. Only live-tier flags reach here — the frozen ones were refused by
    # ``resolve_session`` with a reason, rather than silently ignored.
    if time_limit_hours is not None:
        frozen.settings.time_limit_hours = time_limit_hours
    return replace(frozen, resume=run_id, engine_notes=notes, engine_upgrade=upgrade)


def _session_from_record(
    record: Mapping[str, Any],
    settings: Settings,
    *,
    mode: Literal["paper", "live"] | None,
    run_id: str,
) -> SessionInputs:
    """The session one run **froze**: its frozen settings tier, its mandate, its applied overrides.

    The other half of the pair :func:`_session_from_files` opens — and the reason "what the run
    saw" has exactly one definition: :func:`resume_session` continues a run under it and
    :func:`open_reading` reads a run under it, so a segment and a reading of the same record can
    never be told two different things. Pure decisions apart from the log line: the frozen tier
    comes back through :func:`~noctis.config.rehydrate.rehydrate` (live paths, secrets and
    per-process budgets stay this process's), the mandate through
    :func:`~noctis.research.mandate_from_frozen` (resolved *text*, never a selector re-resolved
    against today's files), and the overrides are the echo lines the record already carries.

    A run that froze **nothing** — history adopted by ``noctis migrate``, or a run predating config
    freezing — restores nothing and is therefore read under the current ``config.yaml`` with no
    overlay at all, which is said out loud rather than inferred from a silently ordinary result.
    """
    from noctis.config.rehydrate import has_frozen_inputs, rehydrate
    from noctis.research import mandate_from_frozen

    if not has_frozen_inputs(record):
        logger.warning(
            "run %s froze no configuration (it predates config freezing, or it is history adopted "
            "by `noctis migrate`), so this segment runs under the current config.yaml and "
            "mandate/ — and freezes them onto the run for every segment after it.",
            run_id,
        )
    resumed = rehydrate(record, settings)
    frozen = record.get("inputs") if has_frozen_inputs(record) else None
    frozen_mandate = frozen.get("mandate") if frozen else None
    active = mandate_from_frozen(frozen_mandate)
    overrides = list(frozen_mandate.get("overrides_applied") or []) if frozen_mandate else []
    return SessionInputs(settings=resumed, mode=mode, mandate=active, overrides=overrides)


def _engine_change_on_resume(
    record: Mapping[str, Any], *, run_id: str, upgrading: bool
) -> tuple[list[str], dict[str, Any] | None]:
    """Apply the engine-change resume policy to one record, before anything opens (story #135).

    The policy itself is pure and lives in :mod:`noctis.observability.engine_change`; this is the
    one place that gives it the two things it cannot compute — the engine **this** checkout is
    (:func:`~noctis.observability.engine_id.fingerprint`) and the clock — and turns its verdict into
    the three outcomes an operator sees:

    * arbiter drift **raises** here, so a refused resume opens no segment and takes no lock;
    * searcher drift is logged now and handed back as event text, because a run that ran two
      engines must say so *in the record*, not only in a terminal nobody kept;
    * ``upgrading`` produces the ``engine_changes`` entry — stamped with the index of the segment
      about to be appended, which is exactly the number of segments the record already carries.

    No drift returns ``([], None)``: nothing logged, nothing recorded, nothing said.
    """
    from datetime import UTC, datetime

    from noctis.observability.engine_change import (
        assert_arbiter_held,
        engine_change,
        engine_notes,
        upgrade_entry,
    )
    from noctis.observability.engine_id import fingerprint
    from noctis.reporting.run_record import utc_iso

    change = engine_change(record, fingerprint())
    assert_arbiter_held(change, run_id=run_id, upgrading=upgrading)
    notes = list(engine_notes(change, upgrading=upgrading))
    for note in notes:
        logger.warning("run %s: %s", run_id, note)
    if not upgrading:
        return notes, None
    return notes, upgrade_entry(
        change,
        at=utc_iso(datetime.now(UTC)),
        segment=len(record.get("segments") or []),
    )


def _adopt_current_config(
    settings: Settings,
    record: dict,
    *,
    mode: Literal["paper", "live"] | None,
    time_limit_hours: float | None,
    run_id: str,
) -> SessionInputs | None:
    """``--rebase-config``: run this segment on the current files and re-freeze them onto the run.

    The middle of the ordinary precedence chain, put back for one session — ``resolve_mandate`` →
    :func:`overlay_mandate` → the CLI's live-tier flags — and then the whole thing re-frozen, so the
    session that adopts a configuration and the record that documents the adoption can never
    describe two different configurations.

    ``None`` means **nothing to adopt**: the current files and the run's frozen config already
    agree, so the caller falls through to an ordinary resume and the epoch stays where it is. A
    bump for a change that never happened would mark the run mixed-config forever, and every
    consumer rendering ``config_epoch > 1`` as "this run changed mid-flight" would be lying.

    That no-op is the reason the candidate configuration is assembled on a **copy**: a mandate may
    bind per-process budgets, which are live tier and therefore never drift, so resolving the
    current mandate to look for drift would otherwise leave its overlay applied to a session that
    adopted nothing — and the same command would run under two different budgets depending on
    whether some unrelated key happened to move. The copy is committed only by adopting it.

    The stamp and the segment index are computed here, at the one point that knows both: the
    appending segment's index is the number of segments the record already carries.
    """
    from datetime import UTC, datetime

    from noctis.config.rehydrate import rebase_inputs
    from noctis.reporting.run_record import utc_iso
    from noctis.research import resolve_mandate

    candidate = settings.model_copy(deep=True)
    active = resolve_mandate(candidate, cli_directive=None, cli_mandate=None)
    overrides = overlay_mandate(candidate, active)
    warn_if_auto_overlay_is_inert(candidate, active)
    rebase = rebase_inputs(
        record,
        candidate,
        mandate=active,
        overrides=overrides,
        execution_mode=mode,
        research_loop=resolve_research_loop(candidate),
        at=utc_iso(datetime.now(UTC)),
        segment=len(record.get("segments") or []),
    )
    if rebase is None:
        logger.info(
            "run %s: --rebase-config found no drift — the current config.yaml and mandate/ still "
            "match what this run froze, so its config_epoch stays at %s",
            run_id,
            _frozen_epoch(record),
        )
        return None
    logger.warning(
        "run %s: adopting the current config.yaml and mandate/ (--rebase-config) — config_epoch "
        "%s → %s, recorded with a before/after entry in the run record",
        run_id,
        _frozen_epoch(record),
        rebase.get("config_epoch"),
    )
    if time_limit_hours is not None:
        candidate.time_limit_hours = time_limit_hours
    return SessionInputs(
        settings=candidate, mode=mode, mandate=active, overrides=overrides, rebase=rebase
    )


def _frozen_epoch(record: Mapping[str, Any]) -> object:
    """The config epoch a record carries, for a message — never re-decided, only read."""
    inputs = record.get("inputs")
    return inputs.get("config_epoch") if isinstance(inputs, Mapping) else None


def _addressed_id(record: dict, address: str) -> str:
    """The id of the run an address resolved to, falling back to the address itself.

    One line, but it is the difference between "cannot resume @nightly-momo" and a message naming
    the run that actually refused — and after a label is reassigned those are not the same run.
    """
    run = record.get("run")
    return str(run.get("run_id") or address) if isinstance(run, dict) else address


def _state_pruned(record: Mapping[str, Any] | None) -> bool:
    """Whether retention has deleted this run's heavy directories, off the record's own marker."""
    run = record.get("run") if isinstance(record, Mapping) else None
    return bool(run.get("state_pruned")) if isinstance(run, Mapping) else False


@dataclass(frozen=True)
class Reading:
    """One process's look at a run: its resolved inputs, its tree, and the record they came from.

    A **value**, not a context manager — nothing here was opened, so nothing has to be closed. It
    is what :func:`open_reading` hands a read-only verb, and it carries exactly what such a verb
    needs to assemble its collaborators: the settings every builder reads (post-overlay for the
    reserved run, the run's own frozen tier for an addressed one), the tree those settings are
    bound to, the address an operator typed, and the record itself for the verbs that quote it.
    """

    inputs: SessionInputs
    # The tree every collaborator built from ``settings`` reads. The reserved ``legacy`` run when
    # nothing was addressed — which may not exist on disk at all, and needs not: a reading creates
    # nothing.
    run_dir: Path
    # What the operator *typed* (an id, ``latest``, a path, ``@label``), never the id it resolved
    # to, so a refusal or an echo is always about a run they can go and look at. ``None`` is the
    # bare form of every reader: the reserved run.
    address: str | None = None
    # The addressed run's record, read once and handed on, so a verb that quotes it does not read
    # ``run.json`` a second time. ``None`` for the reserved run — an unaddressed reading resolves
    # no run and therefore reads no record.
    record: Mapping[str, Any] | None = None

    @property
    def settings(self) -> Settings:
        """The settings this reading resolved — what every collaborator is built from."""
        return self.inputs.settings

    @property
    def mode(self) -> Literal["paper", "live"] | None:
        """The gate's verdict, or ``None`` when this reader never asked for one (D1)."""
        return self.inputs.mode

    @property
    def run_id(self) -> str:
        """The run being read — its tree's name, which is the identity that named the tree."""
        return self.run_dir.name

    @property
    def addressed(self) -> bool:
        """Whether an operator named the run. Exactly ``address is not None`` — one field, so a
        caller can never hold the address and the flag apart (and the legacy-layout guard, whose
        question an address answers, reads this)."""
        return self.address is not None

    @property
    def pruned(self) -> bool:
        """Whether retention removed this run's ``state/``, ``strategies/`` and ``reports/``.

        Only ever true on a reading that asked for one (``readable_pruned=True``): every other
        reader is refused by :class:`RunPrunedError` before it gets a value to ask.
        """
        return _state_pruned(self.record)


def open_reading(
    config_path: str | None = None,
    address: str | None = None,
    *,
    require_gate: bool = False,
    mandate_overlay: bool = True,
    readable_pruned: bool = False,
) -> Reading:
    """Read one run — or the reserved one — the way :func:`resolve_session` opens a session (#292).

    The one entry for a **read-only** verb, and the answer to a whole class of bug: a reader that
    stopped at ``load_settings`` saw the pre-overlay ``promotion.metric`` while the run whose
    champions it was labelling had been steered onto another one. Two shapes, one per address:

    * **unaddressed** — today's chain over the current files (:func:`_session_from_files`:
      settings → gate when asked → mandate overlay), binding nothing, because the reserved
      ``legacy`` tree is what those settings already point at. This is what a run minted right now
      would be told.
    * **addressed** — one resolver turns the operator's string into a run dir (the four forms in
      one fixed order), the record is read **without a lock**, the run's tree is bound onto the
      settings and its frozen inputs come back through :func:`_session_from_record` — the same
      recipe ``--resume`` runs on, so a reading of a run sees what a resume of it would run under.

    Minus the lock, and minus the acting: none of ``assert_resumable``, ``assert_mode_unchanged``,
    the engine-change note or the ``--rebase-config`` adoption runs here. A reader acts on nothing,
    so a **completed** run is readable (terminal is about working a run, not reading one), a run
    another engine is live on is readable (reading must never block or corrupt an engine), and the
    tree is left byte-identical: no lock taken, no record written, nothing created.

    It builds nothing, either — no registry, no lake, no memory, no report. It resolves and binds
    and stops; the command body assembles its collaborators from ``reading.settings``.

    Raises :class:`~noctis.reporting.run_tree.RunNotFoundError` (and its ``RunAmbiguousError``
    subclass) for an address nobody — or too many runs — answer, and for a run whose record is
    there but unreadable; :class:`RunPrunedError` for a run whose tree retention deleted, unless
    ``readable_pruned`` says this verb can read one anyway; and whatever the chain itself refuses
    (:class:`~noctis.config.SafetyGateError` under ``require_gate``,
    :class:`~noctis.research.MandateError` under the overlay).
    """
    if address is None:
        inputs = _session_from_files(
            config_path, require_gate=require_gate, mandate_overlay=mandate_overlay
        )
        return Reading(inputs=inputs, run_dir=Path(inputs.settings.run_dir))
    # Deferred exactly as every other run-tree read in this module is: the composition root is on
    # the ``--help`` path, and the run tree drags in the record schema behind it.
    from noctis.config.settings import bind_run_dir
    from noctis.reporting.run_tree import (
        RUN_RECORD_NAME,
        RunNotFoundError,
        read_record,
        resolve_run_dir,
    )

    settings = load_settings(config_path=config_path)
    mode = resolve_execution_mode(settings) if require_gate else None
    run_dir = resolve_run_dir(Path(settings.runs_dir), address)
    # The listing's tolerant read, turned back into a refusal here: a reading has nothing to read
    # *under* a record it cannot parse. The reason is the record module's own, and it is raised
    # naming the file rather than the address, because "which file, and what is wrong with it" is
    # the whole of what a reader of a broken record needs — and it is the line ``noctis
    # run-record`` has always printed for one (story #297).
    record, note = read_record(run_dir)
    if record is None:
        raise RunNotFoundError(f"{run_dir / RUN_RECORD_NAME}: {note}")
    run_id = _addressed_id(record, run_dir.name)
    if _state_pruned(record) and not readable_pruned:
        raise RunPrunedError(
            f"Run {run_id} was pruned: `noctis run-prune` deleted its reports/ and state/, so "
            f"there is nothing left to read and nothing honest to assemble from. Everything the "
            f"run accumulated is still in its record — `noctis run-record {run_id}`."
        )
    # Bound *before* rehydration, and that order is the point: the run's tree is the live tier
    # (``RUN_IDENTITY`` is never in a record), so binding it here is what makes it survive the copy
    # rehydration takes — and what makes every collaborator afterwards read the addressed run.
    bind_run_dir(settings, run_dir)
    inputs = _session_from_record(record, settings, mode=mode, run_id=run_id)
    return Reading(inputs=inputs, run_dir=run_dir, address=address, record=record)


def overlay_mandate(settings: Settings, mandate: Mandate | None) -> list[str]:
    """Apply one mandate's config overlay, with the gate-unmoved assertion around it.

    **Every** overlay this composition root performs goes through here, so no call site can
    forget the assertion: snapshot the refused settings subtree, apply the patch, assert the
    subtree is byte-identical afterwards. Returns the ``"k=v"`` echo lines ``apply_overrides``
    produced; ``None`` (no mandate) is the same no-op it always was.

    This is belt and braces on top of the deny-by-default classifier, in the spirit of the two
    live-money gates: the classifier decides what a mandate *may* bind, and this proves — after
    the fact, against the live settings object — that nothing else moved. The snapshot is
    derived from :data:`~noctis.config.overlay.REFUSED` itself, so classifying one more path as
    refused extends the assertion with no edit here and the two can never drift.

    It **raises** (:class:`~noctis.config.OverlayError`), never warns: refused settings that
    moved mean the allowlist has a bug, not that the operator mis-typed something, and a run
    that continued would be researching in an arena nobody configured. The message names the
    moved paths and deliberately not their values — the refused subtree carries the API keys.
    """
    from noctis.config.overlay import assert_gates_unmoved, gate_snapshot
    from noctis.research import apply_overrides

    before = gate_snapshot(settings)
    overrides = apply_overrides(settings, mandate)
    assert_gates_unmoved(before, gate_snapshot(settings))
    return overrides


def warn_if_auto_overlay_is_inert(settings: Settings, mandate: Mandate | None) -> None:
    """Warn — once per session — when ``research.mandate: auto`` makes a profile's overlay inert.

    Under ``auto`` the agent chooses its profile *inside* the session, long after settings are
    assembled, so no profile's ``config:`` block can ever reach the overlay. That used to cost one
    knob; now that the overlay carries the whole run configuration (the model, the loop, the spend
    ceilings, the data window) it costs the run's entire steering — and the loss is otherwise
    completely silent: nothing fails, the session simply isn't steered. So scan the catalog at
    startup and name the profiles whose keys will go nowhere, plus the remedy (pin the mandate).

    **Informational, never fatal.** ``auto`` is a valid, shipped configuration and picking a
    profile mid-session against the champion board is the point of it, so this warns and continues
    — the same loud-degradation contract as the LLM-client and reference-loading warnings — rather
    than refusing to start. Pre-selecting the profile before assembly would be a redesign of the
    ``auto`` contract, not a warning.

    "Once per session" is structural, not a flag: this is called from :func:`resolve_session`,
    which every session-assembling entrypoint calls exactly once, before any phase loop starts.
    Which keys count as inert (``promotion.metric`` is deliberately excluded) is the mandate
    module's decision, beside the ``auto`` contract that justifies it.
    """
    from noctis.research import inert_auto_overrides

    if mandate is None or mandate.source != "auto":
        return
    inert = inert_auto_overrides(settings.mandate_dir)
    if not inert:
        return
    named = "; ".join(f"{name} ({', '.join(keys)})" for name, keys in sorted(inert.items()))
    logger.warning(
        "research.mandate: auto — %d profile(s) declare config: keys that will NOT apply this "
        "session: %s. Under auto the agent picks its profile mid-session, long after settings "
        "are assembled, so a profile's config: block never reaches the overlay. Pin the mandate "
        "to make it apply: research.mandate: <profile> in config.yaml, or --mandate <profile> "
        "for one session.",
        len(inert),
        named,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Legacy-layout guard
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LegacyArtifact:
    """A pre-workspace artifact the current configuration would silently abandon."""

    legacy: Path  # where the old (pre-workspace) layout kept it
    configured: Path  # where settings now point


def _guarded_pre_workspace_pairs(settings) -> tuple[tuple[Path, Path], ...]:
    """(legacy, configured) for the four **pre-workspace** artifacts the startup guard checks.

    Anchored next to the config file, the project root in the run-in-place model
    (``_yaml_path().parent``, so ``NOCTIS_CONFIG`` moves the search with it). The configured side
    is read off ``settings``, so each one names where the engine reads that artifact *now* — for
    the three a run owns that is the run's own tree, and the lake stays workspace-level.
    """
    from noctis.config.settings import _yaml_path

    root = _yaml_path().parent
    return (
        (root / "state", Path(settings.state_dir)),
        (root / "data_lake", Path(settings.data.lake_dir)),
        (root / "reports", Path(settings.reports_dir)),
        (root / "MEMORY.md", Path(settings.memory_path)),
    )


def _pre_workspace_pairs(settings) -> tuple[tuple[Path, Path], ...]:
    """Every pre-workspace pair ``migrate`` moves: the four guarded ones plus the strategy tiers.

    The tiers are moved but never guarded — an orphaned ``strategies/__tmp/`` was never a reason
    to refuse a run, since nothing silently reads as empty because of it.
    """
    from noctis.config.settings import _yaml_path
    from noctis.strategies.library import CHAMPIONS_SUBDIR, TMP_SUBDIR, LibraryPaths

    root = _yaml_path().parent
    tiers = LibraryPaths.from_settings(settings)
    return (
        *_guarded_pre_workspace_pairs(settings),
        (root / "strategies" / TMP_SUBDIR, tiers.tmp),
        (root / "strategies" / CHAMPIONS_SUBDIR, tiers.champions),
    )


def _pre_run_scoped_pairs(settings) -> tuple[tuple[Path, Path], ...]:
    """(legacy, configured) for every **pre-run-scoped** artifact — the workspace-level ones.

    Before story #131 the workspace held one ``state/``, one ``reports/``, one ``memory/`` and one
    pair of strategy tiers, shared by every invocation. They are now owned by the run that
    produced them, so an existing operator's copies are adopted into the reserved ``legacy`` run.
    The data lake is deliberately absent: it stays workspace-level and shared by every run.
    """
    from noctis.strategies.library import CHAMPIONS_SUBDIR, TMP_SUBDIR, LibraryPaths

    workspace = Path(settings.workspace_dir)
    tiers = LibraryPaths.from_settings(settings)
    return (
        (workspace / "state", Path(settings.state_dir)),
        (workspace / "reports", Path(settings.reports_dir)),
        (workspace / "memory" / "MEMORY.md", Path(settings.memory_path)),
        (workspace / "qa", Path(settings.qa_dir)),
        (workspace / "strategies" / TMP_SUBDIR, tiers.tmp),
        (workspace / "strategies" / CHAMPIONS_SUBDIR, tiers.champions),
    )


def _orphaned(pairs: tuple[tuple[Path, Path], ...]) -> list[LegacyArtifact]:
    """The pairs whose legacy side exists while the configured side does not — one shared rule.

    Explicitly pointing a knob at the legacy path is honored (the pair is skipped): that is a
    deliberate configuration, not an orphan.
    """
    found: list[LegacyArtifact] = []
    for legacy, configured in pairs:
        if legacy.resolve() == configured.resolve():
            continue  # explicitly configured to the legacy location — intentional
        if legacy.exists() and not configured.exists():
            found.append(LegacyArtifact(legacy=legacy, configured=configured))
    return found


def detect_legacy_layout(settings) -> list[LegacyArtifact]:
    """Find legacy (pre-workspace) artifacts the configured layout would orphan.

    An artifact is flagged when the old default path beside ``config.yaml`` exists, the configured
    location differs, and the configured location does not exist: exactly the naive-upgrade case
    where a run would start against a silently-empty champion board while the real data sits
    abandoned. Callers map a non-empty result to a **refusal** that names ``noctis migrate``;
    ``status`` only warns.
    """
    return _orphaned(_guarded_pre_workspace_pairs(settings))


def detect_unadopted_state(settings) -> list[LegacyArtifact]:
    """Find pre-run-scoped ``workspace/`` state that no run has adopted yet (story #131).

    The younger sibling of :func:`detect_legacy_layout`, and deliberately a **warning** rather
    than a refusal. The difference is what continuing would cost: a pre-workspace artifact is
    genuinely abandoned — nothing else looks there ever again — while un-adopted workspace state
    is sitting safely in the same workspace, and the run that starts beside it is a *new* run with
    its own board, which is correct by design. So the honest instruction is "adopt your history
    into a run when you want it", not "you may not start". One command answers both:
    ``noctis migrate``.
    """
    return _orphaned(_pre_run_scoped_pairs(settings))


def scaffold_init(settings) -> list[str]:
    """Idempotent operator scaffold: local input files + the workspace. Never overwrites.

    Copies each committed template (config, env, mandate) to its local, gitignored name
    when — and only when — the local file doesn't exist yet, and creates the workspace
    root. Returns one human-readable line per action (created / kept / no template),
    which the CLI prints verbatim.
    """
    from noctis.config.settings import _yaml_path

    root = _yaml_path().parent
    lines: list[str] = []
    pairs = (
        (root / "config.example.yaml", root / "config.yaml"),
        (root / ".env.example", root / ".env"),
        (root / "mandate" / "MANDATE.md.example", root / "mandate" / "MANDATE.md"),
    )
    for template, target in pairs:
        if target.exists():
            lines.append(f"kept     {target} (already exists — your edits are safe)")
        elif not template.is_file():
            lines.append(f"skipped  {target} (no template {template.name} here)")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(template.read_bytes())
            lines.append(f"created  {target} (from {template.name})")
    workspace = Path(settings.workspace_dir)
    if workspace.is_dir():
        lines.append(f"kept     {workspace} (already exists)")
    else:
        workspace.mkdir(parents=True, exist_ok=True)
        lines.append(f"created  {workspace} (the one output root)")
    return lines


@dataclass(frozen=True)
class MigrationConflict:
    """A legacy artifact only a human can place: two candidate histories, one destination."""

    legacy: Path
    configured: Path
    reason: str


@dataclass(frozen=True)
class MigrationPlan:
    """What `noctis migrate` would do: clean moves, blocking conflicts, pinned skips."""

    moves: list[LegacyArtifact]
    conflicts: list[MigrationConflict]  # the destination is already taken — refuse
    pinned: list[Path]  # a knob explicitly points at the legacy path — left in place


def plan_migration(settings) -> MigrationPlan:
    """Plan the one-shot move of every legacy artifact to where the engine reads it now.

    Two generations, one plan, because they are one question for the operator ("where did my
    history go?") and deserve one instruction: the **pre-workspace** artifacts beside
    ``config.yaml`` (:func:`_pre_workspace_pairs`) and the **pre-run-scoped** workspace artifacts
    (:func:`_pre_run_scoped_pairs`), which are adopted into the reserved ``legacy`` run. The local
    config never moves — it stays at the root, merely untracked.

    A destination that already exists, or that two legacy copies both claim, is a **conflict**: it
    is refused with a reason rather than resolved by guessing, because the two candidates are two
    different histories and only a human knows which one is theirs. Pure planning: nothing on disk
    changes here.
    """
    pairs = (*_pre_workspace_pairs(settings), *_pre_run_scoped_pairs(settings))
    moves: list[LegacyArtifact] = []
    conflicts: list[MigrationConflict] = []
    pinned: list[Path] = []
    claimed: dict[Path, Path] = {}
    for legacy, configured in pairs:
        if not legacy.exists():
            continue
        target = configured.resolve()
        if legacy.resolve() == target:
            pinned.append(legacy)
        elif configured.exists():
            conflicts.append(
                MigrationConflict(
                    legacy=legacy,
                    configured=configured,
                    reason="both exist — keep one and remove the other, then re-run",
                )
            )
        elif target in claimed:
            conflicts.append(
                MigrationConflict(
                    legacy=legacy,
                    configured=configured,
                    reason=f"two legacy copies claim it ({claimed[target]} is the other) — keep "
                    "one and remove the other, then re-run",
                )
            )
        else:
            claimed[target] = legacy
            moves.append(LegacyArtifact(legacy=legacy, configured=configured))
    return MigrationPlan(moves=moves, conflicts=conflicts, pinned=pinned)


def execute_migration(plan: MigrationPlan) -> None:
    """Perform the planned moves. Call only on a conflict-free plan."""
    import shutil

    for artifact in plan.moves:
        artifact.configured.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(artifact.legacy), str(artifact.configured))


def adopt_run_record(
    settings, *, adopted: int, clock: Callable[[], Any] | None = None
) -> Path | None:
    """Give the run that just adopted legacy state a ``run.json``, so it is a run like any other.

    Adoption produces a run nobody ever *ran*: its champions, account and reports were earned
    before runs existed. It still needs a record — that is what makes it listable by
    ``noctis runs`` and resumable later — so this writes one with **zero segments**, which is the
    honest shape: no process has ever worked this run, and inventing a segment would put a
    fabricated night in the history. The record carries an event saying where its contents came
    from, so the provenance is in the artifact rather than in the operator's memory.

    Idempotent, and never destructive: a run dir that already has a record is left exactly as it
    is, so migrating twice cannot overwrite a real run's history with an adoption stub.
    """
    from datetime import UTC, datetime

    from noctis.reporting.run_record import RecordEvent, RunArtifacts, build, utc_iso
    from noctis.reporting.run_tree import read_record, update_index, write
    from noctis.reporting.run_tree.evidence import read_engine_identity

    run_dir = Path(settings.run_dir)
    if not run_dir.is_dir():
        return None
    record, _ = read_record(run_dir)
    if record is not None:
        return None
    now = (clock or (lambda: datetime.now(UTC)))()
    stamp = utc_iso(now)
    artifacts = RunArtifacts(
        run_id=run_dir.name,
        created_utc=stamp,
        last_active_utc=stamp,
        engine=read_engine_identity(settings.promotion.metric),
        label="default",
        complete=True,
        events=(
            RecordEvent(
                t=stamp,
                kind="info",
                text=f"adopted {adopted} pre-run-scoped artifact(s) into this run by "
                f"`noctis migrate`; its state predates run-scoped state",
            ),
        ),
    )
    write(run_dir, build(artifacts))
    update_index(run_dir.parent, run_dir.name)
    return run_dir / "run.json"


# ─────────────────────────────────────────────────────────────────────────────
# Collaborators
# ─────────────────────────────────────────────────────────────────────────────
def build_memory(settings):
    """The agent's long-term memory store (pure file I/O; LLM upkeep is the distillation step).

    Resolves ``settings.memory_path`` (workspace-derived unless overridden) and, on first
    run, seeds it from the committed ``MEMORY.seed.md`` — found next to the config file,
    like every committed input. The copy happens *before* the store constructs, because
    ``MemoryStore.load`` auto-creates its blank template for a missing file and would win
    the race. No seed ⇒ the blank template; never an error.
    """
    from noctis.config.settings import _yaml_path
    from noctis.memory import MemoryStore

    memory_path = Path(settings.memory_path)
    seed = _yaml_path().parent / "MEMORY.seed.md"
    if not memory_path.exists() and seed.is_file():
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_bytes(seed.read_bytes())
    return MemoryStore(memory_path)


def build_families(settings) -> FamilyRegistry:
    """The one strategy-family hydration: seeds → persisted spec-families → library files.

    The order is the correctness contract, fixed here once: spec-families re-register
    before anything reads the champion board (``champions.json`` stores only ``{family,
    params}``, so a promoted spec-family's class must exist before a champion builds),
    and the library files load last because they are the canonical versions of their
    families — tuned defaults live in the file and must win over any in-repo seed.
    """
    from noctis.strategies.families import FamilyRegistry
    from noctis.strategies.library import LibraryPaths
    from noctis.strategies.library import load_and_register as load_library
    from noctis.strategies.spec import load_and_register as load_specs

    families = FamilyRegistry()
    load_specs(settings.state_dir, families)
    load_library(LibraryPaths.from_settings(settings), families)
    return families


def build_lake(settings, *, require_vendor: bool = False) -> MarketDataLake:
    """Build a MarketDataLake, choosing a vendor from available credentials.

    Without a ``DATABENTO_API_KEY``: read-only commands get a placeholder vendor that
    refuses any fetch; a command that must fetch (``require_vendor=True``) raises
    :class:`MissingVendorKey` instead.
    """
    from noctis.data import MarketDataLake

    vendor: Any  # duck-typed seam: real vendor client or the read-only placeholder
    if settings.databento_api_key:
        from noctis.data.databento_provider import DataBentoVendorClient

        vendor = DataBentoVendorClient(settings.databento_api_key)
    elif require_vendor:
        raise MissingVendorKey("No DATABENTO_API_KEY set — cannot fetch data. Add it to .env.")
    else:
        vendor = _ReadOnlyVendor()
    return MarketDataLake(
        settings.data.lake_dir, vendor, settings.data.budget_usd, settings.session.calendar
    )


class _ReadOnlyVendor:
    """Placeholder vendor for read-only commands (status). Refuses any fetch."""

    def get_cost(self, **_kwargs):  # noqa: D401
        raise RuntimeError("read-only: no vendor configured")

    def fetch_bars(self, **_kwargs):
        raise RuntimeError("read-only: no vendor configured")


def build_event_sink(
    verbose: int,
    *,
    show_reasoning: bool = False,
    secondary: Callable[[Event | str], None] | None = None,
) -> EventSink:
    """The session's ``on_event`` sink — **always a real sink**, never ``None`` (#337).

    One builder, one return type, three shapes of the same seam: a level-aware
    :class:`~noctis.observability.Console` when ``-v``/``-vv``/``--show-reasoning`` asks for a
    view, an :class:`~noctis.observability.EventTee` when a ``secondary`` rides along, and
    otherwise the quiet :data:`~noctis.observability.NULL_SINK` — whose every member is a safe
    default, so a session with nobody watching is *silent* rather than sink-less and no holder
    downstream re-derives what "no sink" means.

    The ``secondary`` (a recorder-style event callable) rides even when nothing is watching: a
    quiet ``--debug`` run tees onto the null primary, so it records every event silently rather
    than not at all. It is typed generically as any event callable, because a secondary is only
    ever *called* — nothing about riding the tee asks it to be a whole sink."""
    if verbose or show_reasoning:
        from noctis.observability import Console

        primary: EventSink = Console(verbose, show_reasoning=show_reasoning)
    else:
        primary = NULL_SINK
    if secondary is None:
        return primary
    from noctis.observability import EventTee

    return EventTee(primary, secondary)


# ─────────────────────────────────────────────────────────────────────────────
# The --debug QA recorder
# ─────────────────────────────────────────────────────────────────────────────
# API keys the config digest must never fold in: the manifest lands under workspace/qa (gitignored),
# but digesting a vendor/LLM credential would still be leaking a secret (AGENTS.md rule 6). The set
# itself lives beside the fields it names (``config.settings.SECRET_FIELDS``) — the run record's
# frozen inputs exclude the same three, and one list is the only way the two can agree forever.
_DIGEST_SECRET_FIELDS = SECRET_FIELDS


def _digest_excluded_fields() -> set[str]:
    """Settings fields the config digest leaves out: the secrets, plus the run's own tree.

    The digest is a **label for grouping runs that share a configuration** (epic #126, D2), so
    anything carrying the run's *identity* has to stay out of it: once a run is opened, the
    run-scoped paths hold its minted id, and folding those in would give every run a unique digest
    and make the label useless for the one job it has. Both halves are named once elsewhere — the
    credentials beside the fields themselves, the run's tree beside the freezing tiers that also
    keep it out of the record — so a path added to either is excluded here with no edit.
    """
    from noctis.config.rehydrate import RUN_IDENTITY

    return set(_DIGEST_SECRET_FIELDS) | set(RUN_IDENTITY)


# ─────────────────────────────────────────────────────────────────────────────
# The per-segment environment probes (story #139)
# ─────────────────────────────────────────────────────────────────────────────
# How long the git probe may block. A reporting fact is never worth a stalled startup: a hung
# `git` on a network filesystem degrades to a null git block with the seam named, exactly like a
# wheel install with no checkout.
_GIT_TIMEOUT_S = 5.0


def build_environment_probes(root: Path | None = None):
    """The real environment probes — the **only** place hardware, git and extras are read.

    ``noctis.observability.environment`` is deliberately free of ``platform``, ``os``, ``socket``
    and ``subprocess``: it shapes what these callables return and nothing more, so a test needs no
    machine, no repository and no subprocess. The reading itself belongs here, beside every other
    collaborator this root assembles.

    ``root`` is the checkout the git state and the lockfile are read from; it defaults to this
    installation's repo root (:func:`~noctis.observability.engine_id.default_root`, the same one
    the engine fingerprint resolves its component paths against, so the record's two identities
    describe the same tree). Injectable so a test can point it at a directory that is not a
    repository and watch both degrade honestly.

    Every probe returns data or ``None`` and none of them raises for an ordinary absence: no
    ``psutil`` (an optional extra by design — the stdlib subset answers what it can), no git
    checkout, no lockfile, none of the optional stacks. ``environment.capture`` turns each absence
    into an explicit ``null`` plus a named degraded seam.
    """
    from noctis.observability.engine_id import default_root
    from noctis.observability.environment import EnvironmentProbes

    base = default_root() if root is None else Path(root)
    return EnvironmentProbes(
        hostname=_probe_hostname,
        os_facts=_probe_os_facts,
        hardware=_probe_hardware,
        versions=_probe_versions,
        git=lambda: _probe_git(base),
        lockfile=lambda: _probe_lockfile(base),
        extras=_probe_extras,
    )


def capture_environment(root: Path | None = None) -> dict:
    """This machine's environment block, ready to ride on the segment this process opens.

    One line of wiring — probes here, shaping there — so the record's per-segment environment has
    exactly one production source and the run store never learns how to read a CPU.
    """
    from noctis.observability.environment import capture

    return capture(build_environment_probes(root)).as_dict()


def _probe_hostname() -> str | None:
    """The raw machine name. Hashed by ``environment.capture`` and never recorded as-is — story
    #129 made the same choice for ``run.lock``, and the two read one hashing function."""
    import socket

    try:
        return socket.gethostname()
    except OSError:  # pragma: no cover - a host that will not name itself
        return None


def _probe_os_facts() -> dict:
    import os
    import platform
    from pathlib import Path

    # Containerisation matters because it bounds the CPU and memory the numbers beside it were
    # produced under. Marker files first (cheap, definitive when present), then the cgroup line a
    # container runtime rewrites; a platform that offers neither answers ``None`` rather than a
    # confident "no".
    container: bool | None = None
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        container = True
    elif os.environ.get("container"):
        container = True
    else:
        try:
            cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="replace")
        except OSError:
            cgroup = ""
        if cgroup:
            container = any(marker in cgroup for marker in ("docker", "kubepods", "containerd"))
    return {
        "system": platform.system() or None,
        "release": platform.release() or None,
        "arch": platform.machine() or None,
        "container": container,
    }


def _probe_hardware() -> dict:
    """CPU, memory and free disk — ``psutil`` when the ``hardware`` extra is installed, the
    stdlib subset otherwise.

    The seam discipline AGENTS.md already applies to ``vectorbt`` and ``databento``: richer
    hardware facts are a nicety, so they may not become a core dependency. Without the extra the
    logical core count and free disk still land, and the rest is honest ``null`` with ``hardware``
    named in ``degraded_seams``.
    """
    import os
    import platform
    import shutil

    facts: dict[str, object] = {
        "model": _stdlib_cpu_model() or platform.processor() or None,
        "cores_physical": None,
        "cores_logical": os.cpu_count(),
        "freq_max_mhz": None,
        "memory_total_bytes": _stdlib_memory_total_bytes(),
        "disk_free_bytes": None,
    }
    try:
        facts["disk_free_bytes"] = shutil.disk_usage(os.getcwd()).free
    except OSError:  # pragma: no cover - a working directory that vanished
        pass
    try:
        # The optional extra, imported only where it is used — never at module scope.
        import psutil

        facts["cores_physical"] = psutil.cpu_count(logical=False)
        facts["cores_logical"] = psutil.cpu_count(logical=True) or facts["cores_logical"]
        facts["memory_total_bytes"] = psutil.virtual_memory().total
        facts["freq_max_mhz"] = getattr(psutil.cpu_freq(), "max", None) or None
    except Exception:
        # An absent extra and a psutil that cannot answer on this kernel are the same event here:
        # the enrichment did not happen. Swallowed rather than raised so the stdlib facts already
        # in hand survive — degrading the whole block to nulls would lose more than it reports.
        return facts
    return facts


def _stdlib_cpu_model() -> str | None:
    """The CPU's marketing name from ``/proc/cpuinfo``, or ``None`` off Linux.

    ``platform.processor()`` is empty on most Linux builds, and the CPU model is the single most
    useful hardware fact for reading two runs' trials-per-hour against each other — so it is worth
    one file read that ``psutil`` does not offer at all.
    """
    from pathlib import Path

    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip() or None
    except (OSError, IndexError):
        return None
    return None


def _stdlib_memory_total_bytes() -> int | None:
    """Total RAM without ``psutil``: ``/proc/meminfo`` on Linux, ``None`` anywhere else."""
    from pathlib import Path

    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        return None
    return None


def _probe_versions() -> dict:
    import platform

    return {"python": platform.python_version(), "noctis": _noctis_version()}


def _probe_git(root: Path) -> dict | None:
    """This checkout's commit, branch, dirty flag and describe — or ``None`` outside a repository.

    Degrading to ``None`` rather than to a block of nulls is deliberate: "there is no git state
    here" (a wheel install, a copied tree) is a different fact from "there is a repository that
    could not answer", and only the first is an ordinary Noctis install.
    """
    commit = _git(root, "rev-parse", "HEAD")
    if commit is None:
        return None
    status = _git(root, "status", "--porcelain")
    return {
        "commit": commit,
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
        "describe": _git(root, "describe", "--tags", "--always", "--dirty"),
    }


def _git(root: Path, *args: str) -> str | None:
    """One git command, bounded and never fatal. ``None`` for anything that did not answer."""
    import subprocess

    try:
        # A fixed argv (never an operator string), bounded, and never checked into a raise.
        done = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _probe_lockfile(root: Path) -> bytes | None:
    """``uv.lock``'s bytes — the whole dependency resolution, pinned by one digest downstream."""
    try:
        return (root / "uv.lock").read_bytes()
    except OSError:
        return None


def _probe_extras() -> dict:
    """``{extra name: version or None}`` over the optional extras the installer knows.

    One list, not two: :data:`noctis.onboarding.EXTRA_MODULES` is what ``noctis setup`` probes and
    what this reports, so "missing extra" and "degraded seam" stay a single notion and the remedy
    the record implies (``uv sync --extra <name>``) is one an operator can type.
    """
    from importlib import metadata
    from importlib.util import find_spec

    from noctis.onboarding import EXTRA_MODULES

    present: dict[str, str | None] = {}
    for extra, module in EXTRA_MODULES.items():
        try:
            installed = find_spec(module) is not None
        except (ImportError, ValueError):  # pragma: no cover - a broken meta path entry
            installed = False
        if not installed:
            present[extra] = None
            continue
        try:
            present[extra] = metadata.version(module.replace("_", "-"))
        except Exception:  # importable but not distribution-installed (a vendored/local module)
            present[extra] = "unknown"
    return present


def _noctis_version() -> str | None:
    """The package literal — informational, never a comparison key (that is the engine version)."""
    from importlib import metadata

    try:
        return metadata.version("noctis")
    except Exception:  # not pip-installed (editable/source tree)
        from noctis import __version__

        return __version__


def open_run_store(
    settings,
    *,
    argv: list[str],
    command: str = "run",
    run_id: str | None = None,
    clock: Callable[[], Any] | None = None,
    label: str | None = None,
    resume: bool = False,
    inputs: SessionInputs | None = None,
):
    """Open this invocation's run — the always-on run identity, minted here and nowhere else.

    Every ``noctis run`` gets a run id, a tree under ``settings.runs_dir``
    (``workspace/runs/<run_id>/``), a liveness lock and one self-describing ``run.json``, whether
    or not ``--debug`` is on. Identity is **minted**, never derived from the configuration: two
    byte-identical configs are two runs unless one explicitly resumes the other (story #131), so
    nothing here hashes settings into an id.

    The id minted here is also the ``--debug`` QA tree's id (``build_recorder`` takes it as an
    argument), so a run has exactly one identity and one tree per artifact instead of two ids
    nobody can correlate.

    **Opening a run also binds its state** (story #131): the run's tree owns its champions, paper
    account, forward ledger, journals, strategy tiers, memory and reports, so this rebinds
    ``settings`` onto that tree (:func:`~noctis.config.settings.bind_run_dir`) the moment the id is
    known. Every collaborator assembled afterwards — ``build_memory``, ``build_families``,
    ``build_registry``, ``build_recorder``, the runtime — therefore reads the *run's* state without
    a single path edit in a command body, and two runs in one workspace cannot contaminate each
    other. The shared data lake is untouched: vendor data is expensive and run-neutral.

    **Opening a run also records what it ran on and under** (story #139): the appending segment
    carries this machine's environment block — hardware, OS, python, git state, lockfile digest,
    extras present and degraded seams — captured through :func:`build_environment_probes`. It is
    per segment, never per run, because a stopped-and-resumed experiment may migrate machines and
    research throughput is CPU-bound: one night's trials-per-hour must not be attributed to
    another night's cores.

    **Opening a run also freezes its config** (story #132): the settings this invocation assembled,
    the mandate as resolved *text* plus the overlay it applied, the resolved models and the data
    provider/dataset, and the gate's verdict are pinned onto the record at creation — and only at
    creation, so every later segment restores them instead of
    re-reading files that may have changed since. ``resume=True`` says this invocation
    is continuing an existing run rather than minting one, which turns an unknown id and a
    ``completed`` run into refusals rather than a surprise new run.

    **What is frozen comes from the session, whole** (story #248): ``inputs`` is the
    :class:`SessionInputs` an entrypoint resolved, and the five fields freezing reads — the gate's
    verdict, the mandate, the overrides it applied, a rebase and an engine upgrade — are read off
    it here rather than unpacked into five kwargs each command body has to keep in step. A tier of
    resume policy added tomorrow is therefore a field on that object, not a sixth kwarg threaded
    through two Typer bodies. ``None`` is the bare form a direct caller (or a test) uses: no
    session was resolved, so no verdict was measured and nothing is adopted.

    ``inputs.rebase`` is the deliberate re-freeze a ``--rebase-config`` resume built (story #134):
    already epoch-bumped and change-stamped by :func:`resume_session`, it **replaces** the inputs
    the record carried instead of being ignored like a fresh freeze would be. Absent (the normal
    case) nothing about freezing changes at all.

    ``inputs.engine_upgrade`` is its twin one layer down (story #135): the ``engine_changes`` entry
    an accepted ``--allow-engine-upgrade`` produced. Given one, the run's engine identity is
    re-frozen onto this process's engine with that entry appended; absent, the run keeps the engine
    it was created under, which is what every resume is compared against.

    Raises :class:`~noctis.reporting.run_tree.RunLockedError` when another engine already holds
    the addressed run — the one failure in this subsystem that is fatal rather than latched.
    """
    from datetime import UTC, datetime

    from noctis.config.rehydrate import freeze_inputs
    from noctis.config.settings import bind_run_dir
    from noctis.reporting.run_record import utc_iso
    from noctis.reporting.run_tree import open_run

    tick = clock or (lambda: datetime.now(UTC))
    # The session's frozen-tier fields, read off the one object that carries them. Without a
    # session there is nothing to freeze but the settings themselves: an unmeasured mode, no
    # steering, nothing adopted.
    rebase = inputs.rebase if inputs is not None else None
    # ``open_run``'s ``inputs=`` is the run record's frozen configuration block — a different thing
    # from the session above, so it is named here rather than built inline against a parameter of
    # the same name: an adopted rebase, or this invocation's own freeze.
    frozen = (
        rebase
        if rebase is not None
        else freeze_inputs(
            settings,
            mandate=inputs.mandate if inputs is not None else None,
            overrides=inputs.overrides if inputs is not None else [],
            execution_mode=inputs.mode if inputs is not None else None,
            research_loop=resolve_research_loop(settings),
            frozen_at=utc_iso(tick()),
        )
    )
    store = open_run(
        Path(settings.runs_dir),
        clock=tick,
        argv=list(argv),
        election_metric=settings.promotion.metric,
        run_id=run_id,
        command=command,
        label=label,
        resume=resume,
        inputs=frozen,
        rebase_config=rebase is not None,
        engine_upgrade=inputs.engine_upgrade if inputs is not None else None,
        # The machine this process is on (story #139), captured through the injected probes above
        # and stamped on THIS segment only: a run that migrates boxes mid-experiment keeps each
        # night's throughput attributed to the hardware that produced it.
        environment=capture_environment(),
    )
    bind_run_dir(settings, store.run_dir)
    return store


def segment_phase_seconds(result) -> dict[str, float]:
    """This segment's per-phase working seconds, read off a ``RuntimeResult`` (story #136).

    The measurement the record turns into the run's cumulative research/trading seconds — summed
    across ``segments[]`` at write time, never carried in memory across a restart. Empty for a
    process that never entered a phase (a startup failure), which the record then reports as "this
    segment measured nothing" rather than as a run that spent zero seconds researching. Duck-typed
    like :func:`segment_counters`: the run store never imports the engine.
    """
    measured = getattr(result, "phase_seconds", None) or {}
    return {str(phase): float(seconds) for phase, seconds in measured.items()}


def segment_counters(result) -> dict[str, int]:
    """This segment's own counters, read off a ``RuntimeResult``.

    Per-segment rather than per-run because throughput is only comparable when it is attributed
    to the process that produced it — a run resumed on another machine, or after a code change,
    must not have one segment's work credited to another's conditions. Duck-typed on purpose: the
    run store never imports the engine.
    """
    return {
        "cycles": int(getattr(result, "cycles_completed", 0) or 0),
        "research_iterations": int(getattr(result, "research_iterations", 0) or 0),
        "research_promotions": int(getattr(result, "research_promotions", 0) or 0),
        "trades": int(getattr(result, "trades", 0) or 0),
    }


def research_segment_counters(summary) -> dict[str, int]:
    """A **research-only** segment's counters, read off one ``ResearchSummary`` (story #137).

    The standalone-session twin of :func:`segment_counters`, and deliberately spelled in the same
    words where it means the same thing (``research_iterations`` / ``research_promotions`` are what
    the loop accumulates from these very fields), so per-segment throughput compares across the two
    verbs. ``sessions`` is the one thing only this path knows: a ``noctis research`` invocation *is*
    one session, where a loop segment holds as many as the night had cycles.

    There is deliberately **no ``trades`` key**: a research session cannot place an order, and the
    record derives ``run.traded`` from that counter — a zero written here would be a claim about
    trading made by a process that never traded. Duck-typed like its twin.
    """
    return {
        "sessions": 1,
        "research_iterations": int(getattr(summary, "iterations", 0) or 0),
        "research_promotions": int(getattr(summary, "promotions", 0) or 0),
    }


def build_recorder(settings, *, argv: list[str], mode: str | None, run_id: str | None = None):
    """Assemble the ``--debug`` QA recorder — the one place the run tree is minted (story #45).

    Prune-on-start first (retention per ``qa.keep_last_runs``), then take the run's id and
    construct a :class:`~noctis.observability.debug.Recorder` under ``settings.qa_dir`` with a UTC
    wall-clock and the manifest fields the recorder cannot know itself: the CLI ``argv``, the run
    ``mode``, a deterministic config digest, and the noctis/python versions. The recorder owns run
    id and the started/stopped/duration stamps; everything else is injected here. The digest is
    taken over the *resolved* settings, minus the fields :func:`_digest_excluded_fields` names —
    the API keys, so a credential can never ride into the report tree, and the run's own tree, so
    the digest stays a label two runs on one configuration share.

    ``run_id`` is the run's own id (:func:`open_run_store`), so the QA tree and the run record
    describe the same run under one name; it defaults to a freshly minted id for callers that
    have no run store, which keeps this function usable on its own.
    """
    import hashlib
    import platform
    from datetime import UTC, datetime

    from noctis.observability.debug import Recorder, new_run_id, prune_qa_dir

    prune_qa_dir(settings.qa_dir, settings.qa.keep_last_runs)

    dump = settings.model_dump_json(exclude=_digest_excluded_fields())
    config_digest = hashlib.sha256(dump.encode("utf-8")).hexdigest()[:12]

    manifest = {
        "argv": list(argv),
        "mode": mode,
        "config_digest": config_digest,
        "versions": {"noctis": _noctis_version(), "python": platform.python_version()},
    }
    return Recorder(
        settings.qa_dir,
        run_id=run_id or new_run_id(),
        clock=lambda: datetime.now(UTC),
        manifest=manifest,
    )


# ─────────────────────────────────────────────────────────────────────────────
# The run segment — one context manager owns open → work → close
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Segment:
    """One process's open stretch of work on a run: its store, its recorder, its event sink.

    The handle :func:`open_segment` yields. It carries what the work needs — the run's store, the
    ``--debug`` recorder (or ``None``), the ``on_event`` sink the runtime and the research session
    take (always a real sink: the quiet :data:`~noctis.observability.NULL_SINK` when nobody is
    watching) — plus read-through conveniences for the four things a command echoes (:attr:`run_id`,
    :attr:`record_path`, :attr:`resumed`, :attr:`prior_runtime_s`), so a command body never reaches
    through to the store for them.

    :meth:`finish` is the **only** mutation: it reports what stopped this segment and what it
    measured, and the band closes with that when the ``with`` block unwinds. Everything else here
    reads. A body that never finishes closes at the ``"startup"`` sentinel — the honest reading of
    a process that fell over before it measured anything.
    """

    inputs: SessionInputs
    command: Literal["run", "research"]
    store: RunStore
    recorder: Recorder | None
    on_event: EventSink = NULL_SINK
    # What ``finish`` reported, held until the band closes the segment. ``None`` means "nothing was
    # reported": the sentinel reason, and — separately, for each measurement — nothing measured.
    _reason: str | None = field(default=None, init=False, repr=False)
    _counters: dict[str, int] | None = field(default=None, init=False, repr=False)
    _phase_seconds: dict[str, float] | None = field(default=None, init=False, repr=False)

    @property
    def run_id(self) -> str:
        """The run this segment belongs to — minted for a fresh run, resolved for a resumed one."""
        return str(self.store.run_id)

    @property
    def record_path(self) -> Path:
        """The one file this run writes."""
        return Path(self.store.record_path)

    @property
    def resumed(self) -> bool:
        """Whether this session continues an existing run — exactly ``inputs.resume is not None``,
        read off the one field that carries the address, so the flag can never disagree with it."""
        return self.inputs.resume is not None

    @property
    def prior_runtime_s(self) -> float:
        """Runtime the run had already accumulated before this segment opened (story #136)."""
        return float(self.store.prior_runtime_s)

    def checkpoint(self, outcome: Any) -> None:
        """Rewrite the record mid-segment — the day loop's ``on_cycle_close`` seam.

        Incremental durability: a multi-week run's evidence is current on disk long before the
        process stops, so a machine that dies at 3am still has last cycle's numbers. Takes the same
        ``RuntimeResult``-shaped outcome the loop hands its seam, and derives the same two
        measurements the close derives, so an interrupted segment and a clean one are counted the
        same way.
        """
        self.store.checkpoint(
            counters=segment_counters(outcome),
            phase_seconds=segment_phase_seconds(outcome),
        )

    def finish(
        self,
        reason: str,
        *,
        outcome: Any = None,
        phase_seconds: Mapping[str, float] | None = None,
    ) -> None:
        """Report what stopped this segment, and what it measured. The band closes with it.

        Counters are derived **by command**, from the outcome the work actually produced: a
        ``RuntimeResult`` under ``run`` (four counters plus the phase timings it measured itself),
        a ``ResearchSummary`` under ``research`` (one session and its two research counters, and
        deliberately no ``trades`` key — a session that cannot place an order must not write a zero
        the record would read as "this run traded nothing"). Research measures its own working
        seconds around the session, so ``phase_seconds`` is the caller's there; given explicitly it
        always wins, since an explicit measurement beats a derived one.

        ``outcome=None`` means **nobody measured**: no counters, no phase seconds, never zeros.
        Called twice, the **first** reason wins and the call is otherwise a no-op — the store's
        close is idempotent and a segment stops once, so a later cleanup path cannot rewrite the
        reason the work chose.
        """
        if self._reason is not None:
            return
        self._reason = reason
        if outcome is None:
            return
        if self.command == "research":
            self._counters = research_segment_counters(outcome)
        else:
            self._counters = segment_counters(outcome)
            self._phase_seconds = segment_phase_seconds(outcome)
        if phase_seconds is not None:
            self._phase_seconds = {str(phase): float(s) for phase, s in phase_seconds.items()}

    def _reported(self) -> tuple[str, dict[str, int] | None, dict[str, float] | None]:
        """What to close the store with: the reported reason, or the ``"startup"`` sentinel."""
        if self._reason is None:
            return "startup", None, None
        return self._reason, self._counters, self._phase_seconds


def _segment_recorder(
    inputs: SessionInputs,
    *,
    command: Literal["run", "research"],
    argv: list[str],
    debug: bool,
    run_id: str,
) -> Any:
    """The ``--debug`` QA recorder for this segment, under the command's own rule — or ``None``.

    Two rules, because the two verbs mean different things by a missing LLM. ``run`` records
    whatever the night does: without a buildable client the legacy proposer/Optuna loop drives
    research, which instruments no funnel, so the recorder is *marked* legacy and renders the
    honesty line instead of a comforting all-zeros funnel (AGENTS.md rule 2). ``research`` without
    a client has no session to run at all, so it records nothing rather than leaving an orphaned
    half-written run tree behind.

    The client is only ever asked about under ``--debug``, exactly as both command bodies asked
    before: a bare run answers no question about an LLM it never needed.
    """
    if not debug:
        return None
    from noctis.research import client_status

    llm_ready = client_status(inputs.settings).ok
    if command == "research" and not llm_ready:
        return None
    recorder = build_recorder(inputs.settings, argv=argv, mode=inputs.mode, run_id=run_id)
    if command == "run" and not llm_ready:
        recorder.mark_legacy_research()
    return recorder


@contextmanager
def open_segment(
    inputs: SessionInputs,
    *,
    command: Literal["run", "research"],
    argv: list[str],
    label: str | None = None,
    debug: bool = False,
    verbose: int = 0,
    show_reasoning: bool = False,
    clock: Callable[[], Any] | None = None,
) -> Iterator[Segment]:
    """Open one run segment, hand it to the work, and close it however the work ends (#249).

    A ``noctis run`` and a ``noctis research`` invocation both open exactly one segment: take the
    lock, mint or resume the run id, freeze the inputs, record every engine-change note, build the
    ``--debug`` recorder, build the event sink, do the work, then stamp the segment with its stop
    reason and counters and release the lock. That lifecycle used to be written twice, by hand, in
    two Typer command bodies — so a lock-release fix could land in one and be missed in the other.
    It is one context manager here instead, and the commands keep only what is theirs: the echoes,
    the exit codes, and the collaborators they drive.

    **The resume decision is read off the session, whole** (D2/D3): ``inputs.resume`` is the
    address an operator typed and ``resume is not None`` is the flag, so no entrypoint can carry
    the two apart, and a tier of resume policy added tomorrow is a field on
    :class:`SessionInputs` rather than another kwarg threaded through two command bodies.
    ``label`` stays a kwarg: it is a nickname for *this* invocation, not a session input.

    **Recording is the band's half of the engine-change note** (D4): every note the resume policy
    found lands on the record here, whether or not a terminal ever printed it — a record is what an
    experiment is judged from months later. Saying it is the command's half, and it says it from
    ``seg.inputs``.

    **Everything after the lock is guarded** (D5): the recorder and the event sink are assembled
    *inside* the try, so a recorder that refuses to build (an unwritable ``qa_dir``, a prune that
    raised) still closes the segment instead of leaving the run locked until the stale-lock
    timeout.

    **Every exit path closes once.** A normal return, a body that reported nothing
    (``reason="startup"``, no counters — "measured nothing", never zeros), an exception, a
    ``typer.Exit`` raised by a command body: all reach the same close, in the same order the
    commands used — recorder first, then the store, which releases the lock.

    The one failure this lets out is a live lock
    (:class:`~noctis.reporting.run_tree.RunLockedError`): two engines writing one run is
    corruption, not degradation. It leaves as the typed error, because this module never imports
    Typer and never exits the process — mapping it to red text and an exit code is the command's
    job, and only the command knows which code.
    """
    store = open_run_store(
        inputs.settings,
        argv=argv,
        command=command,
        run_id=inputs.resume,
        resume=inputs.resume is not None,
        clock=clock,
        label=label,
        inputs=inputs,
    )
    # Against the segment it is true of, so a run that changed engines twice says which night was
    # which. A resume that found no drift records nothing at all — silence is the signal.
    for note in inputs.engine_notes:
        store.note(note)

    recorder = None
    segment: Segment | None = None
    try:
        recorder = _segment_recorder(
            inputs, command=command, argv=argv, debug=debug, run_id=store.run_id
        )
        segment = Segment(
            inputs=inputs,
            command=command,
            store=store,
            recorder=recorder,
            # One sink the work emits through, whatever this session asked for: a level-aware
            # console under -v, the quiet null adapter without one, and under --debug a tee that
            # also feeds the recorder — so a quiet --debug run records silently rather than not
            # at all.
            on_event=build_event_sink(verbose, show_reasoning=show_reasoning, secondary=recorder),
        )
        yield segment
    finally:
        if recorder is not None:
            recorder.close()
        # Closing the segment stamps its stop reason and duration, writes the record one last time
        # and releases the lock — including on the paths that never reached the work, so a run that
        # exits early is still a closed, resumable run rather than a dangling lock.
        reason, counters, phase_seconds = (
            segment._reported() if segment is not None else ("startup", None, None)
        )
        store.close(reason=reason, counters=counters, phase_seconds=phase_seconds)


# ─────────────────────────────────────────────────────────────────────────────
# The agent research session
# ─────────────────────────────────────────────────────────────────────────────
# The episodic memory-distillation default (episodic-research epic #62): when the episodic loop
# is selected and the operator left ``research.memory_distill_every`` at its global-default 0,
# distillation defaults ON at this modest cadence. Applied as a per-session *effective value* on
# the shared settings instance in the loop-selection path (:meth:`ResearchSession.run`) — never a
# change to the class default — so a conversation-loop session's behavior stays bit-identical.
_EPISODIC_DISTILL_DEFAULT = 1

# The context window the episodic briefings assert against when the operator left
# ``research.agent.context_window`` unset. Generous so the build-time fit assertion is effectively
# inert (matching the conversation loop's unlimited history); an operator on a small-context
# backend sets ``context_window`` to engage the real discipline (and, at or below
# ``_EPISODIC_WINDOW_MAX``, the ``auto`` flip).
_EPISODIC_CONTEXT_WINDOW = 128_000

# The ``auto`` flip threshold (#76): a declared ``research.agent.context_window`` at or below this
# selects the episodic driver. 32_768 — confirmed against the real local box (a 32k ``num_ctx``
# backend is exactly the small-context machine the episodic loop was built for), inclusive so the
# canonical noctis-ollama config flips. The evidence gate is the parity harness (docs/parity.md):
# PASS on 2026-07-23 — episodic held verdicts/session at 45% fewer tokens/verdict.
_EPISODIC_WINDOW_MAX = 32_768


def resolve_research_loop(settings) -> str:
    """Which research loop this session runs — ``"conversation"`` | ``"episodic"`` — from
    ``research.agent.loop``.

    ``"conversation"`` and ``"episodic"`` are explicit operator picks. ``"auto"`` (the default)
    is the evidence-gated flip (#76): episodic when the operator declared a
    ``research.agent.context_window`` of at most ``_EPISODIC_WINDOW_MAX`` tokens, conversation
    for larger or unset windows (hosted backends). The one place this decision lives is this
    function, so the entrypoints never learn about it.
    """
    loop = settings.research.agent.loop
    if loop in ("conversation", "episodic"):
        return loop
    window = settings.research.agent.context_window
    return "episodic" if window is not None and window <= _EPISODIC_WINDOW_MAX else "conversation"


def effective_memory_distill_every(settings) -> int:
    """The memory-distillation cadence for this session: the operator's ``memory_distill_every``
    when set, otherwise the episodic default (#62) when the episodic loop is selected, else off.

    Pure — the loop-selection path applies it to the shared settings so the CLOSE-phase
    distillation reads the effective value with no change to the global default (the
    conversation loop keeps ``0`` = off, bit-identical to today)."""
    configured = int(settings.research.memory_distill_every or 0)
    if configured:
        return configured
    if resolve_research_loop(settings) == "episodic":
        return _EPISODIC_DISTILL_DEFAULT
    return 0


def build_fallback_panel_source(settings, lake) -> Callable[[], list[str]]:
    """The episodic MATCH *fallback* fit panel, as a zero-arg source resolved at each MATCH (#110).

    The panel itself is unchanged — the ready :func:`~noctis.engine.runtime.trading_roster` names
    capped at ``research.fit_set_size``, exactly what this root used to precompute once and hand
    over as a frozen list. What changes is *when* it is computed: the driver calls this on the MATCH
    that needs it, so a symbol that joins the lake mid-session (a mandate preflight fetch, a later
    DISCOVER episode) is in the next MATCH's panel instead of being frozen out at assembly time,
    and a session whose screens all match never pays the readiness I/O. Deterministic screening
    still owns the per-thesis fit set; this is only the no-lake-match floor.
    """
    from noctis.engine.runtime import trading_roster

    def resolve() -> list[str]:
        ready = [s for s in trading_roster(settings, lake) if lake.check_symbol_ready(s)]
        return ready[: settings.research.fit_set_size]

    return resolve


@dataclass
class ResearchSession:
    """One assembled agent research session: client + budgets + toolbox, ready to run.

    Built by :func:`build_research_session`; ``noctis research`` and the runtime's RESEARCH
    phase both run exactly this bundle, so their loop kwargs can never drift apart again. The
    loop that actually drives the session — the conversation transcript or the episodic driver —
    is resolved from ``research.agent.loop`` inside :meth:`run`, so both entrypoints follow the
    same selection without a code change.

    The bundle holds the *collaborators the root assembled*, not just the toolbox they were
    handed to (#260): the ``lake`` the run's data seam resolved to, the ``budgets`` the active
    cost profile produced, the resolved ``mandate``. Wiring the loop reads them from here — a
    root that fetched them back out of the toolbox would be trusting a copy for a value it holds.
    """

    settings: Settings
    toolbox: ResearchToolbox
    #: The market-data lake this session researches on — the one the root built, handed to the
    #: toolbox and kept here so the episodic fallback panel resolves over the same lake.
    lake: Any
    client: Any
    budgets: CostProfile
    mandate: Mandate | None
    #: Where this session's observable moments go — always a real sink, defaulting to the quiet
    #: :data:`~noctis.observability.NULL_SINK`, so both loops emit without asking whether one is
    #: attached.
    on_event: EventSink = NULL_SINK

    @property
    def price_table(self):
        """The ``$/Mtok`` table this session's spend is estimated with (story #140).

        Assembled here, from this run's own ``research.pricing`` — one resolution both loops are
        handed, so the session line an operator reads and the run record's spend block can never be
        priced from two different tables.
        """
        from noctis.research.pricing import table_from_config

        return table_from_config(self.settings.research.pricing)

    @property
    def model(self) -> str:
        """The resolved provider/model string this session will drive — the one resolution the
        research seam owns, so the session, the client probe, and the CLI's status line all name
        the same model (post-overlay, when a mandate moved it)."""
        from noctis.research import resolved_research_model

        return resolved_research_model(self.settings)

    def run(self, *, max_iterations: int | None = None, stop_event=None) -> ResearchSummary:
        """Run the session behind the ``research.agent.loop`` selector. ``max_iterations`` falls
        back to the cost-profile budget for either loop."""
        if resolve_research_loop(self.settings) == "episodic":
            # Apply the episodic memory-distillation default as a per-session effective value on
            # the shared settings (never the global default), so CLOSE distills on the episodic
            # cadence while a conversation session stays bit-identical.
            self.settings.research.memory_distill_every = effective_memory_distill_every(
                self.settings
            )
            return self._run_episodic(max_iterations=max_iterations, stop_event=stop_event)
        return self._run_conversation(max_iterations=max_iterations, stop_event=stop_event)

    def _run_conversation(self, *, max_iterations: int | None, stop_event) -> ResearchSummary:
        """The conversation loop — one long tool-use transcript. Unchanged from before the loop
        knob: byte-identical kwargs, so ``auto``/unset selects exactly today's behavior."""
        from noctis.research import run_agent_research

        agent_cfg = self.settings.research.agent
        return run_agent_research(
            toolbox=self.toolbox,
            client=self.client,
            budget_minutes=self.settings.research_time_budget_minutes,
            max_iterations=max_iterations or self.budgets.max_iterations,
            max_tokens=agent_cfg.max_tokens,
            context_window=agent_cfg.context_window,
            stop_event=stop_event,
            web_search=self.budgets.web_search,
            max_web_searches=self.budgets.max_web_searches,
            prefix_trim=self.budgets.prefix_trim,
            on_event=self.on_event,
            mandate=self.mandate,
            price_table=self.price_table,
        )

    def _run_episodic(self, *, max_iterations: int | None, stop_event) -> ResearchSummary:
        """The episodic driver — a deterministic session machine that calls the model only at
        narrow judgment episodes and executes everything else through the gated toolbox. The
        episode runner (which holds the client) and the ledger are assembled here and injected;
        the driver itself never sees the client. Returns the same summary shape as the
        conversation loop, so the runtime and the CLI are untouched."""
        from noctis.research.driver import make_episodes, run_episodic_research
        from noctis.research.episode import EpisodeRunner
        from noctis.research.ledger import SessionLedger

        settings = self.settings
        agent_cfg = settings.research.agent
        runner_kwargs: dict[str, Any] = {}
        if agent_cfg.max_tokens:
            runner_kwargs["max_tokens"] = agent_cfg.max_tokens
        runner = EpisodeRunner(
            client=self.client,
            retries=agent_cfg.episode_retries,
            on_event=self.on_event,
            **runner_kwargs,
        )
        ledger = SessionLedger(settings.state_dir)
        context_window = agent_cfg.context_window or _EPISODIC_CONTEXT_WINDOW
        # The three judgment episodes: formulate, decide, and the no-lake-match discover (#112).
        formulate, decide, discover = make_episodes(
            runner=runner,
            toolbox=self.toolbox,
            ledger=ledger,
            mandate=self.mandate,
            context_window=context_window,
        )
        return run_episodic_research(
            toolbox=self.toolbox,
            ledger=ledger,
            formulate=formulate,
            decide=decide,
            # The DISCOVER episode a ``no_lake_match`` MATCH spends before accepting the fallback
            # panel (#112). It rides the same runner as the other two, so its completions count
            # against the one ``max_episodes`` budget below.
            discover=discover,
            fallback_panel_source=build_fallback_panel_source(settings, self.lake),
            budget_minutes=settings.research_time_budget_minutes,
            max_episodes=max_iterations or self.budgets.max_iterations,
            completions=lambda: runner.completions,
            stop_event=stop_event,
            mandate_source=self.mandate.source if self.mandate else None,
            # The two inputs of the session-start mandate-symbol preflight (#111) — and the same
            # ``history_days`` window a DISCOVER fetch covers (#112). The driver reads no settings,
            # so they arrive as values: the resolved mandate's declared symbols (already upper-cased
            # and deduped at parse) and the existing ``data.history_days`` lookback — no mandate ⇒
            # an empty sequence ⇒ a strict no-op preflight.
            mandate_symbols=self.mandate.symbols if self.mandate else (),
            history_days=settings.data.history_days,
            models={"driver": self.model, "coder": agent_cfg.coder_model},
            sweep_trials=self.budgets.sweep_trials,
            on_event=self.on_event,
            price_table=self.price_table,
        )


def _build_coder_client(settings):
    """The dedicated strategy-authoring ("coder") client for ``research.agent.coder_model``, or
    ``None`` — inert in this story, threaded into the toolbox for a follow-up to consume.

    Unset (the default) ⇒ ``None``: the session driver authors full strategy source itself, and
    session assembly is unchanged. Set ⇒ a second, stateless per-model client is built alongside
    the driver via the shared :func:`~noctis.research.client_for` constructor. Thinking flips ON
    here (``research.agent.coder_thinking``, default on) because authoring — the scenario-window
    and warmup arithmetic — is the reasoning-heavy sub-task (#17); it is a *deliberate*, budgeted
    decision (``deliberate=True``), so even a Sonnet coder reasons, while the driver loop's own
    thinking pin is untouched (its cost stays bounded by the Class-B ``max_author_calls`` budget).
    If that client can't be built (its provider's key or the ``[llm]`` extra is missing) the
    degradation is loud, never silent: warn and fall back to ``None`` (driver-authored mode), so
    the session still assembles — the same graceful-degradation contract as the rest of the LLM
    seam, never a mid-session failure.

    The coder's sampling knobs (#222) ride the same builder: unset (the default) they ask for
    nothing, so the coder's request is today's; a set knob is only sent where the provider seam
    declares the capability, so the same mandate pointed at an Anthropic coder drops levers that
    provider has no parameter for instead of erroring."""
    from noctis.research import client_for

    agent = settings.research.agent
    coder_model = agent.coder_model
    if not coder_model:
        return None
    coder = client_for(
        settings,
        coder_model,
        thinking=agent.coder_thinking,
        deliberate=True,
        temperature=agent.coder_temperature,
        seed=agent.coder_seed,
    )
    if coder is None:
        logger.warning(
            "coder_model %r is configured but no coder client could be built (its provider's "
            "API key or the [llm] extra is missing) — assembling in driver-authored mode; the "
            "session driver will write full strategy source itself. See docs/configuration.md.",
            coder_model,
        )
    return coder


def _build_coder_fallback_client(settings):
    """The PAID coder-fallback client for ``research.agent.coder_fallback_model`` (story #72), or
    ``None`` — the counted escalation target a spent local author falls back to.

    Escalation is a fallback FROM local authoring, so this is built only when BOTH a local
    ``coder_model`` and a ``coder_fallback_model`` are configured; either unset ⇒ ``None`` (no
    escalation path, and no wasted client). Built stateless beside the local coder through the
    shared :func:`~noctis.research.client_for` constructor, on its OWN thinking dial
    (``coder_fallback_thinking``, default off — #98): the fallback is the strong model, and the
    reasoning dial tuned for weak local coders broke the escalation insurance in the field (a
    thinking sonnet-5 timed out and thinking-truncated every file), so by default the escalated
    call spends its whole output ceiling on the file. ``deliberate=True`` still marks the opt-in
    (``coder_fallback_thinking: on``) as the budgeted coder decision, so even a Sonnet fallback
    then reasons. If that client can't be built (its provider's key or the ``[llm]`` extra is
    missing) the degradation is loud, never silent: warn and fall back to ``None``, so the session
    still assembles and a failed local author is simply skipped as today — the same
    graceful-degradation contract as :func:`_build_coder_client`, never a mid-session failure.
    Bounded per session by ``research.agent.max_escalations`` (0 = never escalate)."""
    from noctis.research import client_for

    agent = settings.research.agent
    if not agent.coder_model or not agent.coder_fallback_model:
        return None
    fallback = client_for(
        settings,
        agent.coder_fallback_model,
        thinking=agent.coder_fallback_thinking,
        deliberate=True,
        # One coder sampling policy per session (#222) — the escalated coder samples the way the
        # local one was told to, capability-gated per provider like every other lever.
        temperature=agent.coder_temperature,
        seed=agent.coder_seed,
    )
    if fallback is None:
        logger.warning(
            "coder_fallback_model %r is configured but no fallback client could be built (its "
            "provider's API key or the [llm] extra is missing) — assembling with no escalation "
            "path; a failed local author will be skipped as today. See docs/configuration.md.",
            agent.coder_fallback_model,
        )
    return fallback


def build_research_session(
    *,
    settings,
    lake,
    registry,
    families: FamilyRegistry,
    memory,
    mandate: Mandate | None = None,
    rules: PromotionRules | None = None,
    on_event: EventSink = NULL_SINK,
) -> ResearchSession | None:
    """Assemble one agent research session, or ``None`` when no LLM client is buildable
    (no key for the configured provider / the ``[llm]`` extra missing) — the caller decides
    whether that means an error (CLI) or the legacy-loop fallback (runtime).

    ``on_event`` is the session's one output seam, handed to the toolbox and to the session
    alike; a caller with nobody watching passes nothing and gets the quiet
    :data:`~noctis.observability.NULL_SINK` — silence is an adapter here, never an absence."""
    from noctis.champions.promotion import PromotionRules
    from noctis.research import ResearchToolbox, build_llm_client, resolve_budgets
    from noctis.strategies.library import LibraryPaths, prune_stale_drafts

    # Working-tier housekeeping (story #56): sweep stale, still-undecided drafts out of
    # __tmp/ into __tmp/archive/ *before* the toolbox constructs. The toolbox's init loads and
    # registers the library, so pruning first guarantees no session ever observes a stale
    # corpse mid-assembly. This is session assembly, so it runs regardless of which research
    # path (agent or legacy) the caller ends up choosing. Bounded by research.draft_ttl_hours;
    # None/0 is a no-op. Pure housekeeping — never a verdict or a gate (AGENTS.md rule 2).
    # Prior art: prune_qa_dir in build_recorder.
    archived = prune_stale_drafts(
        LibraryPaths.from_settings(settings).tmp,
        ttl_hours=settings.research.draft_ttl_hours,
    )
    if archived:
        logger.info(
            "pruned %d stale working-tier draft(s) before research assembly: %s",
            len(archived),
            ", ".join(archived),
        )

    client = build_llm_client(settings)
    if client is None:
        return None
    toolbox = ResearchToolbox(
        settings=settings,
        lake=lake,
        registry=registry,
        families=families,
        memory=memory,
        rules=rules if rules is not None else PromotionRules.from_settings(settings),
        mandate_source=mandate.source if mandate else None,
        mandate=mandate,
        coder_client=_build_coder_client(settings),
        coder_fallback_client=_build_coder_fallback_client(settings),
        on_event=on_event,
    )
    return ResearchSession(
        settings=settings,
        toolbox=toolbox,
        lake=lake,
        client=client,
        budgets=resolve_budgets(settings.research),
        mandate=mandate,
        on_event=on_event,
    )
