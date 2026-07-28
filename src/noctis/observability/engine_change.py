"""The engine-change resume policy — the arbiter must not move mid-experiment (story #135).

A run outlives the process that started it, so ``noctis run --resume`` can find a different engine
than the one the run was created under (somebody pulled, somebody edited a gate). The policy splits
on **who changed: the judge, or the searcher** — deliberately the same line the CI ratchet enforces
(:mod:`noctis.observability.engine_ratchet`), read through the same one classifier
(:func:`~noctis.observability.engine_id.tier_of`) over the same one constant.

* **Arbiter drift** (the components that decide what passes and what a number means): **refuse**.
  Champions crowned under one set of gates and champions crowned under another were never
  comparable, and inside a *single* run that is worse than across two: the run's own board would
  hold a mixture nobody can rank. So this raises rather than preferring a side.
* **The escape hatch** — ``--allow-engine-upgrade`` — overrides that refusal, and is never
  invisible: the epoch moves, an entry naming every component that moved is appended to the record,
  and the run is flagged ``mixed_engine`` for good.
* **Searcher drift** (how candidates are found, what the model is told, the shipped profiles, the
  seed library, the memory seed, the record schema): **warn, record, proceed**. Improving the
  searcher must not invalidate an experiment whose arbiter held still.
* **No drift**: silence. A policy that logs something every time is a policy operators learn to
  ignore, and then the one warning that mattered is the one nobody read.

This is **evidence and policy, never a gate** (AGENTS.md rule 2): it decides whether a run may
*continue*, and reaches nothing that decides what passes. Like :mod:`noctis.config.rehydrate` — the
same shape one tier down, for the configuration rather than the code — every function here is pure:
a record and a computed fingerprint in, a verdict (or a record entry) out. No clock, no config, no
I/O; the stamp and the segment index arrive as arguments from the one caller that knows them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from noctis.observability.engine_id import (
    ARBITER_COMPONENTS,
    EngineFingerprint,
    Tier,
    tier_of,
)

__all__ = [
    "ACCEPTED_BY",
    "ENGINE_EPOCH",
    "ComponentChange",
    "EngineChange",
    "EngineChangeError",
    "assert_arbiter_held",
    "engine_change",
    "engine_notes",
    "tier_of",
    "upgrade_entry",
]

# The flag that accepts an engine change, spelled once and quoted in every message about it.
ACCEPTED_BY = "--allow-engine-upgrade"

# The record's engine-identity contract version, and the twin of ``rehydrate.CONFIG_EPOCH``: a run
# starts at 1 and only a deliberate upgrade moves it, so ``engine_epoch > 1`` always means "this
# run's arbiter changed mid-flight, and here is where".
ENGINE_EPOCH = 1


class EngineChangeError(RuntimeError):
    """This run may not continue under this engine — the refusal, raised before anything opens.

    The twin of :class:`~noctis.config.rehydrate.RehydrationError`: a disagreement between a record
    and this process that cannot be resolved by preferring a side. Lifted only by
    :data:`ACCEPTED_BY`, which records what it accepted.
    """


@dataclass(frozen=True)
class ComponentChange:
    """One behavioural component that moved: its tier, both digests, and the files it covers.

    ``files`` is the component's **allowlisted inputs** as this process sees them, not a diff: a
    run record freezes digests per component, not per file, so which of the five prompt modules
    moved cannot be known from it — while *what to go and look at* can, and that is what a warning
    an operator can act on has to name.
    """

    component: str
    tier: Tier
    frozen: str | None
    current: str | None
    files: tuple[str, ...] = ()

    def line(self) -> str:
        """``prompts (searcher): 14eb169506a6b5aa -> 3ba3e0bf1c97134f`` — one greppable line."""
        return (
            f"{self.component} ({self.tier}): {self.frozen or 'null'} -> {self.current or 'null'}"
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "tier": self.tier,
            "from": self.frozen,
            "to": self.current,
        }


@dataclass(frozen=True)
class EngineChange:
    """How the engine this process runs differs from the one a run froze at creation.

    Structured, not rendered, for the reason :class:`~noctis.config.rehydrate.ConfigDrift` is: the
    refusal, the warning and the recorded entry are all built from this one value, so what an
    operator is told and what the record says can never be two different stories. Falsy when the
    engine held still.
    """

    components: tuple[ComponentChange, ...] = ()
    frozen_epoch: int = ENGINE_EPOCH
    frozen_version: int | None = None
    current_version: int | None = None

    def __bool__(self) -> bool:
        return bool(self.components)

    @property
    def arbiter(self) -> tuple[ComponentChange, ...]:
        """The moved components that bind comparability — the ones that refuse a resume."""
        return tuple(change for change in self.components if change.tier == "arbiter")

    @property
    def searcher(self) -> tuple[ComponentChange, ...]:
        """The moved components that only change how candidates are found."""
        return tuple(change for change in self.components if change.tier == "searcher")


def engine_change(record: Mapping[str, Any], current: EngineFingerprint) -> EngineChange:
    """Compare the engine a run froze at creation with the one this process computed.

    Component by component, never as one opaque hash: "a prompt was reworded" and "a gate moved"
    are not the same news, and a single hash could only say that *something* differs — which would
    force either refusing every resume after any edit, or trusting none of them.

    Two nulls are not drift (neither side could identify the component, so nothing is *known* to
    have moved) — the same missing-input rule :func:`~noctis.observability.engine_id.compare` takes.
    A record carrying no readable engine identity yields no change at all: history adopted by
    ``noctis migrate`` froze no engine, and stranding exactly the history that path exists to
    preserve would be the wrong answer.
    """
    engine = _engine(record)
    frozen = engine.get("fingerprint")
    if not isinstance(frozen, Mapping):
        return EngineChange(current_version=current.engine_version)
    computed = current.digests()
    components = tuple(
        ComponentChange(
            component=name,
            tier=tier_of(name),
            frozen=_digest(frozen, name),
            current=_digest(computed, name),
            files=_files(current, name),
        )
        for name in sorted(set(frozen) | set(computed))
        if _known(frozen, name) != _known(computed, name)
    )
    return EngineChange(
        components=components,
        frozen_epoch=_epoch(engine),
        frozen_version=_version(engine),
        current_version=current.engine_version,
    )


def assert_arbiter_held(change: EngineChange, *, run_id: str, upgrading: bool = False) -> None:
    """Refuse a resume the run's own arbiter moved under, unless it is being upgraded on purpose.

    The one refusal this module makes, and it is checked before a segment is opened, before a lock
    is taken and before a line of banner is printed — a run that may not continue must be told so
    at the start rather than after a night's work is attributed to a mixture of engines.

    ``upgrading`` is the operator saying :data:`ACCEPTED_BY`. It lifts the refusal and nothing else:
    the change still has to be recorded (:func:`upgrade_entry`), and the run still carries
    ``mixed_engine`` afterwards.
    """
    if upgrading or not change.arbiter:
        return
    arbiter = ", ".join(sorted(ARBITER_COMPONENTS))
    raise EngineChangeError(
        f"run {run_id} was created under a different engine: {_named(change.arbiter)} "
        f"moved. Those components ({arbiter}) decide what passes and what a number means, so "
        "champions crowned under the old ones and champions crowned under the new ones were never "
        "comparable — and accumulating both inside ONE run is worse than comparing across two. "
        f"So this refuses rather than mix them:\n"
        + "\n".join(f"  {component.line()}" for component in change.arbiter)
        + "\nRestore the engine this run was created under (its digests are in the record), start "
        f"a new run under the current one, or accept the change deliberately with {ACCEPTED_BY} — "
        "which bumps engine_epoch, records which components moved, and flags the run mixed_engine "
        "for good."
    )


def engine_notes(change: EngineChange, *, upgrading: bool = False) -> tuple[str, ...]:
    """The events this change is worth putting on the record — empty when the engine held still.

    One note for the searcher tier (component, both digests, and the files to go and look at) and,
    when an upgrade was accepted, one for that. Deliberately **empty on no drift**: silence is the
    signal that nothing moved, and a policy that always says something teaches operators to skip
    the line where it eventually matters.
    """
    notes: list[str] = []
    if change.searcher:
        notes.append(
            "the engine changed since this run was created: "
            f"{_named(change.searcher)} moved in the searcher tier — how candidates are found, not "
            "what judges them — so the accumulated results stay comparable and this resume "
            "proceeds: "
            + "; ".join(
                f"{component.line()} [{', '.join(component.files)}]"
                for component in change.searcher
            )
        )
    if upgrading and change.arbiter:
        notes.append(
            f"engine upgrade accepted ({ACCEPTED_BY}): {_named(change.arbiter)} moved — the "
            "components that decide what passes and what a number means. This run's champions were "
            "crowned under two engines from here on, so it is flagged mixed_engine and its "
            "comparable key follows the new engine: "
            + "; ".join(component.line() for component in change.arbiter)
        )
    return tuple(notes)


def upgrade_entry(change: EngineChange, *, at: str, segment: int) -> dict[str, Any] | None:
    """The ``engine_changes`` entry an accepted upgrade appends — or ``None`` for a no-op.

    The twin of :func:`~noctis.config.rehydrate.rebase_inputs`'s ``config_change`` entry, and for
    the same reason: a run whose engine changed mid-flight must say so **and say where**, or every
    comparison built on it is false. Every component that moved is named with its tier and both
    digests, so a reader learns what was accepted without recomputing anything.

    **No arbiter drift, no bump.** :data:`ACCEPTED_BY` exists to lift one refusal; passing it when
    the judge never moved is a documented no-op rather than a cosmetic epoch bump that would mark
    the run as engine-changed forever. Searcher drift alone is recorded as an *event*, which is
    what "warn and proceed" means.

    Pure: ``at`` arrives as an already-formatted stamp and ``segment`` as the index the appending
    segment will take, because nothing here reads a clock or a record.
    """
    if not change.arbiter:
        return None
    return {
        "at": at,
        "segment": segment,
        "from_epoch": change.frozen_epoch,
        "to_epoch": change.frozen_epoch + 1,
        "from_engine_version": change.frozen_version,
        "to_engine_version": change.current_version,
        "components": [component.as_record() for component in change.components],
        "accepted_by": ACCEPTED_BY,
    }


# ── rendering and reading, both tolerant ───────────────────────────────────────────────────


def _named(changes: Sequence[ComponentChange]) -> str:
    """``gates, prompts`` — the moved components, in the order the comparison sorted them."""
    return ", ".join(change.component for change in changes)


def _engine(record: Mapping[str, Any]) -> Mapping[str, Any]:
    engine = record.get("engine")
    return engine if isinstance(engine, Mapping) else {}


def _digest(digests: Mapping[str, Any], name: str) -> str | None:
    """One component's digest as a value — ``None`` for absent, unknown or non-string."""
    digest = digests.get(name)
    return digest if isinstance(digest, str) else None


def _known(digests: Mapping[str, Any], name: str) -> tuple[bool, str | None]:
    """The same digest, paired with whether this side knew the component at all.

    The pair is what makes "present on one side only" count as drift while two nulls do not:
    neither side could identify the component, so nothing is *known* to have moved.
    """
    return (name in digests, _digest(digests, name))


def _files(current: EngineFingerprint, name: str) -> tuple[str, ...]:
    component = current.components.get(name)
    return tuple(component.files) if component is not None else ()


def _epoch(engine: Mapping[str, Any]) -> int:
    """The epoch a record is at, defaulting to the first — a record that lost the key still bumps
    to a *larger* number rather than restarting the count."""
    epoch = engine.get("engine_epoch")
    return epoch if isinstance(epoch, int) and not isinstance(epoch, bool) else ENGINE_EPOCH


def _version(engine: Mapping[str, Any]) -> int | None:
    version = engine.get("engine_version")
    return version if isinstance(version, int) and not isinstance(version, bool) else None
