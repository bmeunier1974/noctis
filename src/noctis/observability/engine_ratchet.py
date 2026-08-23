"""The CI ratchet — behavioural drift cannot land silently.

:mod:`noctis.observability.engine_id` computes what the engine *is*; this module is the check
that the repo's committed statement of it — ``engine_fingerprint.json`` at the repo root — still
matches, and that a change to the **arbiter** came with a declared version bump.

This file holds the **rule** and nothing else: the record, the comparison, ``--write`` and the
report are the shared mechanics in :mod:`noctis.observability.ratchet`, which every ratchet runs
on, and the rule below is the :class:`~noctis.observability.ratchet.RatchetSpec` plus one judging
function it is given. What moved is decided by
:func:`~noctis.observability.engine_id.compare` — the one null rule — so this check and the resume
policy can never disagree about the same edit.

**The tier split, which is the whole design.** The line is the one
:data:`~noctis.observability.engine_id.ARBITER_COMPONENTS` already draws, and it is read through
:func:`~noctis.observability.engine_id.tier_of` — the single classifier this check and the resume
policy (:mod:`noctis.observability.engine_change`) both call. Never restated here: two copies of
that set would eventually disagree, and the disagreement would be silent.

* **Arbiter tier** (the components that decide what passes and what a number means): drift
  **fails**. This is the one change that invalidates every stored champion comparison, so it can
  never land without ``ENGINE_VERSION`` moving with it.
* **Searcher tier** (how candidates are found, what the model is told, the shipped profiles, the
  seed library, the memory seed, the record schema): drift **warns and passes**, naming the
  component and the files. Improving the searcher must not invalidate an experiment whose
  arbiter held still — and a ratchet that fires on a docstring edit gets disabled, which is
  worse than no ratchet at all.

**The precise rule**, in the order it is evaluated:

1. No readable committed record → **fail**. It is the baseline; without it nothing is checked.
2. The record declares another ``ENGINE_VERSION`` than this tree → **fail** as stale. A bump is
   only half the deal: the record has to be regenerated in the same PR so the diff shows a
   reviewer exactly which behavioural surface moved.
3. An arbiter component's digest differs → **fail**, naming the component and the files that
   moved. With the version unchanged that is undeclared arbiter drift; with the version bumped
   the record simply was not regenerated. Either way the fix is the same two steps.
4. Only searcher components differ → **warn**, exit zero. The record is stale in the honest,
   tolerated way; the warning names what to regenerate, and blocks nothing.

So "a stale record is what the contributor is told to fix" and "searcher drift warns and passes"
compose: staleness is always *reported* and always names the regeneration command; whether it is
*blocking* is decided by which tier moved.

**And the rule ``--write`` adds**: regenerating is the fix this module recommends in every message
it prints, and it rewrites *every* component at once — so it cannot also be the way case 3's
undeclared arbiter move gets recorded, or the ratchet would only hold for contributors who read
the failure before typing the command it printed. ``--write`` therefore evaluates the check first
and, on **arbiter drift with the versions in agreement**, writes nothing and exits 1, printing the
same bump-or-restore guidance plus its refusal. Every other case still regenerates in one command
— searcher-only drift, an arbiter move whose bump is declared, no drift at all, and a missing or
unreadable record (there is nothing to compare against, and that is how the baseline is created).
An arbiter move must arrive *declared*; the tool will not launder one.

The comparison itself is pure — two records in, a structured result out — so every scenario is
testable by fingerprinting a temp tree, editing one file, and fingerprinting it again. The I/O
(reading the committed file, writing it, printing, the exit code) lives in :func:`main`, which
``scripts/engine_fingerprint.py`` and the pre-commit hook call, and the refusal is a decision in
that write path rather than a change to what a comparison *means*.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from noctis.observability import ratchet
from noctis.observability.engine_id import (
    COMPONENT_PATHS,
    ENGINE_VERSION,
    fingerprint,
    tier_of,
)
from noctis.observability.ratchet import (
    Judgement,
    Moved,
    RatchetResult,
    RatchetSpec,
    Status,
    WriteOutcome,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Repo-root-relative, so the record sits beside the tree it describes and a reviewer sees it move
# in the same diff as the behaviour it records.
RECORD_PATH = "engine_fingerprint.json"

# The one command that refreshes it. Named in every failure and warning this module emits, and
# documented in docs/development.md — a check that does not say how to fix it gets disabled.
REGENERATE_COMMAND = "uv run python scripts/engine_fingerprint.py --write"

RECORD_KIND = "noctis.engine_fingerprint"

# Where the declared version lives, quoted in the failure so nobody has to go looking.
_VERSION_SOURCE = "src/noctis/observability/engine_id.py"

# The one thing ``--write`` will not do. Printed under the ordinary bump-or-restore guidance, in
# place of it, because "regenerate the record" is precisely the advice being refused here.
_WRITE_REFUSAL = (
    "refusing to regenerate: --write cannot be the way an undeclared arbiter move gets recorded"
)


def _identity(root: Path) -> dict[str, tuple[str | None, str | None]]:
    """Each component's digest and why it is null, from the one module that computes them."""
    computed = fingerprint(root)
    return {
        name: (component.digest, component.note) for name, component in computed.components.items()
    }


def _header(root: Path) -> dict[str, Any]:
    """The declared comparison key, which is what makes a record a *statement* and not a hash.

    ``root`` is unused on purpose: the version is **declared** by this checkout's constant, never
    read out of the tree being stated — which is exactly why a record can be stale.
    """
    return {"engine_version": ENGINE_VERSION}


def _judge(
    computed: Mapping[str, Any], committed: Mapping[str, Any], moved: tuple[Moved, ...]
) -> Judgement:
    """The tier rule and the ``ENGINE_VERSION`` agreement — the module docstring, as code."""
    computed_version = _version_of(computed)
    recorded_version = _version_of(committed)
    drifts = tuple(move.tagged(tier_of(move.name)) for move in moved)
    arbiter = tuple(drift for drift in drifts if drift.tag == "arbiter")
    searcher = tuple(drift for drift in drifts if drift.tag == "searcher")

    problems: list[str] = []
    if recorded_version != computed_version:
        problems.append(
            f"stale record: it declares ENGINE_VERSION {recorded_version}, this tree declares "
            f"{computed_version} ({_VERSION_SOURCE})"
        )
    if arbiter:
        names = ", ".join(drift.name for drift in arbiter)
        if recorded_version == computed_version:
            problems.append(
                f"arbiter drift with no ENGINE_VERSION bump: {names}. A change here invalidates "
                f"every stored champion comparison — bump ENGINE_VERSION in {_VERSION_SOURCE} in "
                "this PR, or restore the behaviour"
            )
        else:
            problems.append(
                f"arbiter drift recorded under an older ENGINE_VERSION: {names}. The bump is "
                "there; the record was not regenerated with it"
            )
    if searcher:
        names = ", ".join(drift.name for drift in searcher)
        problems.append(f"searcher drift (advisory, never blocking): {names}")

    status: Status = "ok"
    if arbiter or recorded_version != computed_version:
        status = "fail"
    elif searcher:
        status = "warn"
    return Judgement(
        status=status,
        problems=tuple(problems),
        # The strict tier first, so a report reads what blocks before what is advisory.
        drifts=(*arbiter, *searcher),
        # Arbiter drift whose bump *is* already there is a record that was simply not regenerated
        # yet, and regenerating it is the fix — so only agreement on the version is undeclared.
        refuse_write=bool(arbiter) and recorded_version == computed_version,
    )


def _version_of(record: Mapping[str, Any]) -> int | None:
    version = record.get("engine_version")
    return version if isinstance(version, int) else None


SPEC = RatchetSpec(
    title="engine fingerprint ratchet",
    record_path=RECORD_PATH,
    record_kind=RECORD_KIND,
    regenerate_command=REGENERATE_COMMAND,
    entries_key="components",
    asset_paths=COMPONENT_PATHS,
    identity=_identity,
    header=_header,
    judge=_judge,
    write_refusal=_WRITE_REFUSAL,
    prog="engine_fingerprint",
    description=(
        "Check (or regenerate) the committed engine fingerprint record. Drift in an arbiter "
        "component without an ENGINE_VERSION bump fails; searcher-tier drift warns and "
        "passes, naming the component and the files that moved. --write regenerates every "
        "other case in one command, and refuses that one: an arbiter move must arrive "
        "declared."
    ),
    write_help=(
        f"Regenerate {RECORD_PATH} from the tree. Commit it in the same PR. Refuses (writing "
        "nothing, exit 1) on arbiter drift with no ENGINE_VERSION bump."
    ),
)


def build_record(root: Path | None = None) -> dict[str, Any]:
    """The record for the tree at ``root``: the declared version, digests, per-file digests."""
    return ratchet.build_record(SPEC, root)


def load_record(root: Path | None = None) -> dict[str, Any] | None:
    """The committed record, or ``None`` when it is absent or unreadable as a record."""
    return ratchet.load_record(SPEC, root)


def write_record(root: Path | None = None) -> Path:
    """Write the record for ``root``, unconditionally — the plain writer a baseline needs."""
    return ratchet.write_record(SPEC, root)


def regenerate(root: Path | None = None) -> WriteOutcome:
    """Regenerate the committed record — unless that would silently record an arbiter move."""
    return ratchet.regenerate(SPEC, root)


def check(root: Path | None = None) -> RatchetResult:
    """Recompute the tree's record and compare it with the committed one."""
    return ratchet.check(SPEC, root)


def compare_records(
    computed: Mapping[str, Any], committed: Mapping[str, Any] | None
) -> RatchetResult:
    """Compare a freshly computed record with the committed one. Pure: no files, no clock."""
    return ratchet.compare_records(SPEC, computed, committed)


def main(argv: Sequence[str] | None = None) -> int:
    """The entrypoint CI, pre-commit and contributors run. Returns the process exit code."""
    return ratchet.main(SPEC, argv)
