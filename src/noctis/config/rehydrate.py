"""Config freezing and rehydration — the three tiers a resumed run reads its settings from.

A run outlives the process that started it (epic #126): ``noctis run --resume <run_id>`` appends a
segment to an existing run and keeps accumulating research hours, trials, champions and P&L into the
same record. That is only *meaningful* if the run carries its own configuration, because the
alternative — reading ``config.yaml`` and ``mandate/`` fresh at every resume — lets an edit made on
Tuesday retroactively change what Monday's results meant. So a run's config is **frozen at
creation** and restored from the record on every later segment.

This module is the whole decision, and it is deliberately :mod:`noctis.config.overlay`'s neighbour
rather than part of the run store: it is a statement about *settings*, it owns no run, no record
writer and no clock, and :func:`rehydrate` is pure — ``(record, live_settings) -> Settings``, no
I/O. Rebuilding a ``Settings`` from its sources would re-read the environment, ``.env`` and the YAML
file (exactly the files a resume must ignore for frozen keys), so nothing here ever constructs one:
the live object is copied and the frozen values are merged into the copy through the overlay's own
validating applier, so a value restored from a record faces exactly the validators ``config.yaml``
faces.

**The three tiers.**

* :data:`FROZEN` — everything that decides what the accumulated results *mean*: research,
  promotion, backtest, trading, risk, universe, session, champion count, data provider/dataset,
  the research time budget. Restored from the record; the current files are ignored for these.
  The mandate is frozen alongside them as **resolved text** plus its applied overlay, not as a
  selector — freezing ``profile:aggressive`` would freeze nothing at all, since the file behind
  that name can be rewritten tomorrow.
* :data:`LIVE` — always from the current process. The **secrets** (redacted out of the record, so
  the live ``.env`` is their only possible source — that is a feature: a record is shareable
  without leaking credentials), **every path/workspace knob** (a run may resume on a machine with
  different absolute paths), and the **per-process budgets** (``time_limit_hours``, the QA and
  observability knobs, the vendor spend ceiling), which bound one night rather than one experiment.
* :data:`REFUSED` — ``mode`` and ``allow_live``. Never written to the record, never restored. The
  safety gate (:mod:`noctis.config.gate`) re-resolves fresh at every process start, and a frozen
  mode that disagrees with the freshly resolved one is a **hard error**
  (:func:`assert_mode_unchanged`), so a paper run's results can never acquire live segments and a
  record can never resurrect a live mode (AGENTS.md rule 1).

**Two of those tiers are derived, not re-listed.** The refused pair is exactly the overlay's
live-money refusals, the path knobs are exactly its state/IO refusals, and the secrets are the one
set ``Settings`` names — all read through :func:`~noctis.config.overlay.refused_paths`. Only the
per-process budgets are written out here, and :data:`FROZEN` is then the **complement**: every leaf
the live model has that is not live and not refused. So a config field added tomorrow freezes by
default, which is the safe direction — it keeps meaning attached to results — and a new path knob
or credential joins the live tier the moment it is classified in the table it already has to be
classified in.

**Drift is normal and silently fine here: frozen wins.** Inspecting drift, and deliberately
adopting it, is story #134; nothing in this module compares the record against the current files.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel

from noctis.config import overlay
from noctis.config.overlay import OverlayError
from noctis.config.settings import _RUN_SCOPED_SUBPATHS, SECRET_FIELDS, Settings

__all__ = [
    "FROZEN",
    "LIVE",
    "REFUSED",
    "RUN_IDENTITY",
    "RehydrationError",
    "assert_mode_unchanged",
    "classify",
    "freeze_inputs",
    "frozen_digest",
    "has_frozen_inputs",
    "rehydrate",
]

Tier = Literal["frozen", "live", "refused", "unknown"]

# The record's frozen-inputs contract version. Bumped only by a deliberate re-freeze
# (``--rebase-config``, story #134); a run whose config changed mid-flight must say so.
CONFIG_EPOCH = 1


class RehydrationError(RuntimeError):
    """A record that cannot be resumed under this process's configuration.

    Raised for the one disagreement that is never resolvable by preferring a side: a frozen
    execution mode that differs from the freshly resolved one. Every other difference between the
    record and the current files is ordinary drift, and frozen simply wins.
    """


def _leaf_paths(model: type[BaseModel], prefix: str = "") -> frozenset[str]:
    """Every leaf dotted path in a settings model — the same walk the overlay's ratchet does."""
    leaves: set[str] = set()
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            leaves |= _leaf_paths(annotation, f"{prefix}{name}.")
        else:
            leaves.add(f"{prefix}{name}")
    return frozenset(leaves)


_LEAVES = _leaf_paths(Settings)

# Bounds on one *process*, not on one experiment. A run is stopped each morning and resumed each
# night, so how long tonight may last, how much of the vendor budget tonight may spend, and how
# much QA/observability noise tonight keeps are the operator's call now — not a decision the run
# made weeks ago. Everything else about spend is frozen: the model, the loop, the Class-B session
# ceilings all shape what the results mean.
_PER_PROCESS: frozenset[str] = frozenset(
    {
        # The wall-clock ceiling on this invocation (`--time-limit-hours` bounds one night).
        "time_limit_hours",
        # Retention for the --debug QA tree, and the live heartbeat cadence: display and
        # housekeeping, read by no decision path at all.
        "qa.keep_last_runs",
        "observability.heartbeat_polls",
        # The vendor spend ceiling for THIS process. Cumulative spend is the record's business;
        # what tonight may burn is the operator's.
        "data.budget_usd",
    }
)

# Tier 3 — never written to a record, never restored. Derived from the overlay's live-money
# refusals, so the pair of gates that must come from two independent sources is named once.
REFUSED: frozenset[str] = overlay.refused_paths(overlay.LIVE_MONEY)

# Tier 2 — always the current process's. Two of the three parts are derived (see the module
# docstring); only the per-process budgets are a judgment written out above.
LIVE: frozenset[str] = (
    overlay.refused_paths(overlay.STATE_IO) | SECRET_FIELDS | _PER_PROCESS
) & _LEAVES

# Tier 1 — what the accumulated results mean. The complement, so a new field freezes by default.
FROZEN: frozenset[str] = _LEAVES - LIVE - REFUSED

# The paths that name the run's own tree. Live tier like every other path, and additionally kept
# out of the record entirely: they are the run's *identity*, the record already sits inside them,
# and a run whose record quoted the reserved default it was assembled under (settings are built
# before a run exists) would be quoting a directory that was never its own. The ``--debug``
# manifest's config digest excludes the same set for the neighbouring reason — a digest that folded
# in the run id would be unique per run and useless as a grouping label.
RUN_IDENTITY: frozenset[str] = frozenset(_RUN_SCOPED_SUBPATHS) | {"run_dir"}


def classify(path: str) -> Tier:
    """Which freezing tier one dotted settings path belongs to.

    Total over *any* string: a path this ``Settings`` model does not have is ``"unknown"``, so a
    stale key in an old record is skipped rather than smuggled into a live section rebuild.
    """
    if path in REFUSED:
        return "refused"
    if path in LIVE:
        return "live"
    if path in FROZEN:
        return "frozen"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Freezing — what the record carries
# ─────────────────────────────────────────────────────────────────────────────
def freeze_inputs(
    settings: Settings,
    *,
    mandate: Any = None,
    overrides: Sequence[str] = (),
    execution_mode: str | None = None,
    frozen_at: str,
) -> dict[str, Any]:
    """The record's ``inputs`` block: this run's configuration, pinned at creation.

    ``resolved`` is the settings dump **minus the secrets and minus the refused pair** — a record
    is meant to be shareable, and a mode is a verdict the gate re-reaches every start rather than a
    value a file may hand back. The path knobs stay in the dump as evidence of where this run ran;
    they are live tier, so :func:`rehydrate` never restores one.

    ``execution_mode`` is the **gate's verdict** for this run, not the ``mode`` setting: it is
    recorded so a later segment can prove it is continuing the same kind of run
    (:func:`assert_mode_unchanged`), and it can never be read back into ``Settings``.

    ``mandate`` is duck-typed on the research package's ``Mandate`` (this is the config layer; it
    does not import research). Its **resolved text** is frozen, with a digest beside it, together
    with the overlay it applied — so what the agent was told, and what that steering moved, are
    both pinned against a profile file that may be rewritten tomorrow.

    Pure: ``frozen_at`` arrives as an already-formatted stamp, because nothing here reads a clock.
    """
    frozen = _frozen_values(settings)
    return {
        "config_epoch": CONFIG_EPOCH,
        "frozen_at_utc": frozen_at,
        "execution_mode": execution_mode,
        "mandate": _frozen_mandate(mandate, overrides),
        "settings": {
            "digest": _digest(frozen),
            "resolved": _resolved_dump(settings),
            "frozen_keys": sorted(FROZEN),
            "live_keys": sorted(LIVE),
            "refused_keys": sorted(REFUSED),
        },
    }


def frozen_digest(settings: Settings) -> str:
    """A short label for "these two runs mean the same thing": a digest over the frozen tier only.

    Deliberately narrower than the ``--debug`` manifest's config digest, which labels the whole
    resolved configuration: a path knob or a per-process budget moving must not change this one, or
    resuming on a second machine would look like a different experiment.
    """
    return _digest(_frozen_values(settings))


def has_frozen_inputs(record: Mapping[str, Any]) -> bool:
    """Whether this record froze a configuration at all.

    ``False`` for a run adopted from pre-run-scoped state (story #131), which has a record but was
    never *started* by a process, and for any record written before freezing existed. Resuming one
    is a normal resume against the current files — the alternative, refusing, would strand exactly
    the history the adoption path exists to preserve.
    """
    return isinstance(record.get("inputs"), Mapping)


# ─────────────────────────────────────────────────────────────────────────────
# Rehydration — what a resumed process reads
# ─────────────────────────────────────────────────────────────────────────────
def rehydrate(record: Mapping[str, Any], live_settings: Settings) -> Settings:
    """The settings a resumed segment runs under: frozen tier from the record, live tier from here.

    Pure, and the epic's config-freezing decision in one function. The live object is **copied**
    (never mutated, never rebuilt from its sources) and the record's frozen values are merged into
    the copy through :func:`~noctis.config.overlay.revalidated`, so the frozen tier is validated by
    exactly the validators ``config.yaml`` gets. Paths, secrets and per-process budgets are simply
    not merged, so they stay whatever this process resolved.

    A key the record carries that this ``Settings`` no longer has is skipped: a record must stay
    readable by the Noctis that resumes it, and one removed knob is not a reason to strand a
    multi-week experiment. A record with no frozen inputs restores nothing
    (:func:`has_frozen_inputs`).
    """
    resumed = live_settings.model_copy(deep=True)
    frozen = {
        path: value
        for path, value in _flattened(_frozen_resolved(record)).items()
        if classify(path) == "frozen"
    }
    if not frozen:
        return resumed
    try:
        rebuilt = overlay.revalidated(resumed, frozen)
    except OverlayError as exc:  # a hand-edited or foreign record, re-typed for the resume path
        raise RehydrationError(
            f"this run's frozen configuration could not be restored: {exc}"
        ) from exc
    for name, value in rebuilt.items():
        setattr(resumed, name, value)
    _assert_gates_unmoved(live_settings, resumed)
    return resumed


def _assert_gates_unmoved(live: Settings, resumed: Settings) -> None:
    """Prove, after the fact, that rehydration left the live-money gates exactly where the gate
    put them — the same belt-and-braces the composition root wraps every mandate overlay in.

    The classification already makes it impossible (:data:`REFUSED` is never merged), so this can
    only fire on a bug in this module. That is precisely when it matters: a record that could move
    ``mode`` or ``allow_live`` would be a second source for a decision that must have exactly two
    independent ones, and rule 1 is not a thing to hold by construction *and* leave unchecked.
    """
    moved = sorted(path for path in REFUSED if getattr(live, path) != getattr(resumed, path))
    if moved:  # pragma: no cover - unreachable unless the tiers themselves are wrong
        raise RehydrationError(
            "rehydration moved a live-money safety gate — this is a bug in the freezing tiers, "
            f"not in the record: {', '.join(moved)}. The gate is re-resolved at every process "
            "start and is never restored."
        )


def assert_mode_unchanged(record: Mapping[str, Any], execution_mode: str) -> None:
    """Refuse a resume whose frozen execution mode differs from the freshly resolved one.

    The safety gate is never rehydrated — it re-resolves from ``config.yaml`` + ``ALLOW_LIVE`` at
    every process start, and this checks its *verdict* against the one the run's earlier segments
    ran under. A paper run's results must not acquire live segments, and a live run's must not
    acquire paper ones: either way the accumulated numbers would silently mix two different kinds
    of evidence. So this raises rather than preferring a side. A record that froze no mode (an
    adopted run) never blocks a resume.
    """
    frozen = _inputs(record).get("execution_mode")
    if frozen is None or frozen == execution_mode:
        return
    raise RehydrationError(
        f"this run's segments were produced in {frozen!r} mode, but this process resolved "
        f"{execution_mode!r}. The safety gate re-resolves at every start and is never restored "
        "from a record, so the two must already agree: a run's accumulated results may not mix "
        f"paper and live segments. Start a new run, or restore the {frozen!r} configuration "
        "(config mode + the ALLOW_LIVE environment gate) and resume again."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────
def _inputs(record: Mapping[str, Any]) -> Mapping[str, Any]:
    inputs = record.get("inputs")
    return inputs if isinstance(inputs, Mapping) else {}


def _frozen_resolved(record: Mapping[str, Any]) -> Mapping[str, Any]:
    settings = _inputs(record).get("settings")
    resolved = settings.get("resolved") if isinstance(settings, Mapping) else None
    return resolved if isinstance(resolved, Mapping) else {}


def _resolved_dump(settings: Settings) -> dict[str, Any]:
    """The settings as data, minus everything a record may not carry: the secrets, the two gates,
    and the run's own tree. What is left — including the workspace-level paths — is evidence of
    the configuration this run was started under; the live tier of it is never restored."""
    excluded = set(SECRET_FIELDS) | set(REFUSED) | set(RUN_IDENTITY)
    return json.loads(settings.model_dump_json(exclude=excluded))


def _frozen_values(settings: Settings) -> dict[str, Any]:
    """``{dotted path: value}`` for the frozen tier, read off one settings object."""
    dump = _flattened(json.loads(settings.model_dump_json()))
    return {path: dump[path] for path in sorted(FROZEN) if path in dump}


def _flattened(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """A nested settings dump as dotted leaves. A list value is a leaf (``universe`` is one)."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flattened(value, prefix=f"{path}."))
        else:
            flat[path] = value
    return flat


def _digest(values: Mapping[str, Any]) -> str:
    """``sha256[:12]`` over a canonical rendering — sorted keys, so it depends on values only."""
    payload = json.dumps(values, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _frozen_mandate(mandate: Any, overrides: Sequence[str]) -> dict[str, Any] | None:
    """One resolved mandate as record data: its text, its provenance, its applied overlay."""
    if mandate is None:
        return None
    text = str(getattr(mandate, "text", ""))
    return {
        "source": str(getattr(mandate, "source", "")),
        "summary": str(getattr(mandate, "summary", "")),
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "symbols": [str(symbol) for symbol in getattr(mandate, "symbols", []) or []],
        "config_overrides": copy.deepcopy(dict(getattr(mandate, "config_overrides", {}) or {})),
        "overrides_applied": [str(line) for line in overrides],
        "references": [
            {
                "path": str(getattr(reference, "path", "")),
                "text": str(getattr(reference, "text", "")),
                "bytes": len(str(getattr(reference, "text", "")).encode("utf-8")),
                "sha256": hashlib.sha256(
                    str(getattr(reference, "text", "")).encode("utf-8")
                ).hexdigest(),
            }
            for reference in getattr(mandate, "references", []) or []
        ],
    }
