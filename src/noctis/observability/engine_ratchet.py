"""The CI ratchet — behavioural drift cannot land silently.

:mod:`noctis.observability.engine_id` computes what the engine *is*; this module is the check
that the repo's committed statement of it — ``engine_fingerprint.json`` at the repo root — still
matches, and that a change to the **arbiter** came with a declared version bump.

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

**And the rule ``--write`` adds** (:func:`regenerate`): regenerating is the fix this module
recommends in every message it prints, and it rewrites *every* component at once — so it cannot
also be the way case 3's undeclared arbiter move gets recorded, or the ratchet would only hold for
contributors who read the failure before typing the command it printed. ``--write`` therefore
evaluates the check first and, on **arbiter drift with the versions in agreement**, writes nothing
and exits 1, printing the same bump-or-restore guidance plus its refusal. Every other case still
regenerates in one command — searcher-only drift, an arbiter move whose bump is declared, no drift
at all, and a missing or unreadable record (there is nothing to compare against, and that is how
the baseline is created). An arbiter move must arrive *declared*; the tool will not launder one.

The comparison itself is pure — two records in, a structured result out — so every scenario is
testable by fingerprinting a temp tree, editing one file, and fingerprinting it again. The I/O
(reading the committed file, writing it, printing, the exit code) lives in :func:`main`, which
``scripts/engine_fingerprint.py`` and the pre-commit hook call, and the refusal is a decision in
that write path rather than a change to what a comparison *means*.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from noctis.observability.engine_id import (
    COMPONENT_PATHS,
    Tier,
    default_root,
    file_digest,
    fingerprint,
    tier_of,
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

Status = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class ComponentDrift:
    """One component that moved: its tier, the digests either side, and the files that moved."""

    component: str
    tier: Tier
    files: tuple[str, ...]
    recorded: str | None
    computed: str | None

    def line(self) -> str:
        recorded = self.recorded or "null"
        computed = self.computed or "null"
        return f"{self.component} ({self.tier}): {recorded} -> {computed}"


@dataclass(frozen=True)
class RatchetResult:
    """The verdict: a status, why, and exactly which components and files moved."""

    status: Status
    problems: tuple[str, ...] = ()
    arbiter_drift: tuple[ComponentDrift, ...] = ()
    searcher_drift: tuple[ComponentDrift, ...] = ()
    recorded_version: int | None = None
    computed_version: int | None = None

    @property
    def ok(self) -> bool:
        """Whether CI passes. A warning is visible, not blocking."""
        return self.status != "fail"

    @property
    def undeclared_arbiter_drift(self) -> bool:
        """The arbiter moved and the declared version does not say so — what ``--write`` refuses.

        Read *off* the verdict, never folded into it: another reading of the same four cases, not a
        fifth status, so ``--check`` still reports exactly what it always did. Arbiter drift whose
        bump *is* already there is a record that was simply not regenerated yet, and regenerating it
        is the fix — so only the case where both sides declare the same version is undeclared.
        """
        return bool(self.arbiter_drift) and self.recorded_version == self.computed_version

    def report(self) -> str:
        """The human-readable verdict CI prints: what moved, in which files, and the fix."""
        if self.status == "ok":
            advice = "the committed record matches this tree"
        else:
            advice = f"regenerate the record: {REGENERATE_COMMAND}"
        return "\n".join([*self.findings(), f"  {advice}"])

    def findings(self) -> list[str]:
        """The header, the problems and the files that moved — everything but the closing advice.

        Split out because :meth:`WriteOutcome.report` prints the same findings and then
        *contradicts* that closing advice ("regenerate the record" is exactly what it is refusing),
        and one renderer of a verdict is better than two that drift apart.
        """
        lines = [f"{self.status.upper()}  engine fingerprint ratchet ({RECORD_PATH})"]
        lines += [f"  {problem}" for problem in self.problems]
        for drift in (*self.arbiter_drift, *self.searcher_drift):
            lines.append(f"  {drift.line()}")
            lines += [f"      {rel}" for rel in drift.files]
        return lines


@dataclass(frozen=True)
class WriteOutcome:
    """What ``--write`` did: the record it wrote, or the verdict it refused to record.

    Exactly one of the two is meaningful — ``written`` is ``None`` on a refusal, and a refusal
    always carries the verdict it refused on, because the contributor needs to see *which* arbiter
    component moved to decide between bumping and restoring.
    """

    verdict: RatchetResult
    written: Path | None

    @property
    def ok(self) -> bool:
        """Whether the regeneration happened. A refusal is the one non-zero exit ``--write`` has."""
        return self.written is not None

    def report(self) -> str:
        """One line on success; the verdict's findings plus the refusal when nothing was written."""
        if self.written is not None:
            return f"wrote {RECORD_PATH} ({self.written})"
        return "\n".join([*self.verdict.findings(), f"  {_WRITE_REFUSAL}"])


def build_record(root: Path | None = None) -> dict[str, Any]:
    """The record for the tree at ``root``: the declared version, digests, per-file digests.

    Per-file digests are what let a drift report name the file that moved instead of every file
    the component happens to cover.
    """
    base = default_root() if root is None else Path(root)
    fp = fingerprint(base)
    components: dict[str, Any] = {}
    for name, rel_paths in COMPONENT_PATHS.items():
        component = fp.components[name]
        components[name] = {
            "digest": component.digest,
            "note": component.note,
            "files": {
                rel: file_digest(base.joinpath(*rel.split("/"))) for rel in sorted(rel_paths)
            },
        }
    return {
        "kind": RECORD_KIND,
        "regenerate_with": REGENERATE_COMMAND,
        "engine_version": fp.engine_version,
        "components": components,
    }


def load_record(root: Path | None = None) -> dict[str, Any] | None:
    """The committed record, or ``None`` when it is absent or unreadable as a record."""
    path = _record_path(root)
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(loaded, dict) or not isinstance(loaded.get("components"), dict):
        return None
    return loaded


def write_record(root: Path | None = None) -> Path:
    """Write the record for ``root``, unconditionally. Sorted keys, so its diff reads cleanly.

    The plain writer: it states the tree as it is and asks no questions, which is what creating a
    baseline in a fresh tree needs. The *policy* about when regenerating is allowed lives one level
    up in :func:`regenerate`, so the write path can refuse without this function — or
    :func:`build_record` — ever having to know about the previous record.
    """
    path = _record_path(root)
    record = build_record(root)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def regenerate(root: Path | None = None) -> WriteOutcome:
    """Regenerate the committed record — unless that would silently record an arbiter move.

    ``--write`` is the one-command fix every failure and warning here names, and it rewrites
    *every* component at once. That is the loophole it has to close: a PR that moves a searcher
    component (the common case) is told to run this, and if it also absorbed an undeclared arbiter
    digest on the way, the ratchet's whole promise would come down to whether the contributor read
    the failure before typing the command it printed. So the write path evaluates the check first
    and declines exactly the case the check exists to catch — arbiter drift while the recorded and
    computed ``ENGINE_VERSION`` agree — leaving the tree checkable, and failing.

    Every other case still regenerates in one command: searcher-only drift, an arbiter move whose
    bump *is* declared (the record just had not caught up), no drift at all, and — the reason the
    guard cannot simply be "never write over arbiter drift" — a missing or unreadable committed
    record, which is how the baseline gets created in the first place.
    """
    base = default_root() if root is None else Path(root)
    verdict = check(base)
    if verdict.undeclared_arbiter_drift:
        return WriteOutcome(verdict=verdict, written=None)
    return WriteOutcome(verdict=verdict, written=write_record(base))


def check(root: Path | None = None) -> RatchetResult:
    """Recompute the tree's record and compare it with the committed one."""
    base = default_root() if root is None else Path(root)
    return compare_records(build_record(base), load_record(base))


def compare_records(
    computed: Mapping[str, Any], committed: Mapping[str, Any] | None
) -> RatchetResult:
    """Compare a freshly computed record with the committed one. Pure: no files, no clock.

    See the module docstring for the rule; this function is that rule, in that order.
    """
    computed_version = _version_of(computed)
    if committed is None:
        return RatchetResult(
            status="fail",
            problems=(
                f"no readable committed record at {RECORD_PATH} — it is the baseline this check "
                "compares against, so nothing can be verified without it",
            ),
            computed_version=computed_version,
        )

    recorded_version = _version_of(committed)
    drifts = _drifts(computed, committed)
    arbiter = tuple(drift for drift in drifts if drift.tier == "arbiter")
    searcher = tuple(drift for drift in drifts if drift.tier == "searcher")

    problems: list[str] = []
    if recorded_version != computed_version:
        problems.append(
            f"stale record: it declares ENGINE_VERSION {recorded_version}, this tree declares "
            f"{computed_version} ({_VERSION_SOURCE})"
        )
    if arbiter:
        moved = ", ".join(drift.component for drift in arbiter)
        if recorded_version == computed_version:
            problems.append(
                f"arbiter drift with no ENGINE_VERSION bump: {moved}. A change here invalidates "
                f"every stored champion comparison — bump ENGINE_VERSION in {_VERSION_SOURCE} in "
                "this PR, or restore the behaviour"
            )
        else:
            problems.append(
                f"arbiter drift recorded under an older ENGINE_VERSION: {moved}. The bump is "
                "there; the record was not regenerated with it"
            )
    if searcher:
        moved = ", ".join(drift.component for drift in searcher)
        problems.append(f"searcher drift (advisory, never blocking): {moved}")

    status: Status = "ok"
    if arbiter or recorded_version != computed_version:
        status = "fail"
    elif searcher:
        status = "warn"
    return RatchetResult(
        status=status,
        problems=tuple(problems),
        arbiter_drift=arbiter,
        searcher_drift=searcher,
        recorded_version=recorded_version,
        computed_version=computed_version,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """The entrypoint CI, pre-commit and contributors run. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="engine_fingerprint",
        description=(
            "Check (or regenerate) the committed engine fingerprint record. Drift in an arbiter "
            "component without an ENGINE_VERSION bump fails; searcher-tier drift warns and "
            "passes, naming the component and the files that moved. --write regenerates every "
            "other case in one command, and refuses that one: an arbiter move must arrive "
            "declared."
        ),
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--check",
        action="store_true",
        help="Compare the tree with the committed record (the default action).",
    )
    action.add_argument(
        "--write",
        action="store_true",
        help=(
            f"Regenerate {RECORD_PATH} from the tree. Commit it in the same PR. Refuses (writing "
            "nothing, exit 1) on arbiter drift with no ENGINE_VERSION bump."
        ),
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Repo root to check (defaults to this checkout).",
    )
    args = parser.parse_args(argv)
    root = None if args.root is None else Path(args.root)

    if args.write:
        outcome = regenerate(root)
        print(outcome.report())
        return 0 if outcome.ok else 1

    result = check(root)
    print(result.report())
    return 0 if result.ok else 1


def _record_path(root: Path | None) -> Path:
    base = default_root() if root is None else Path(root)
    return base / RECORD_PATH


def _version_of(record: Mapping[str, Any]) -> int | None:
    version = record.get("engine_version")
    return version if isinstance(version, int) else None


def _drifts(
    computed: Mapping[str, Any], committed: Mapping[str, Any]
) -> tuple[ComponentDrift, ...]:
    """Every component whose digest differs, with the files that moved under it.

    Two nulls are not drift (neither side could identify the component, so nothing is known to
    have moved) — the same rule :func:`noctis.observability.engine_id.compare` takes.
    """
    computed_components = _components(computed)
    committed_components = _components(committed)
    drifts: list[ComponentDrift] = []
    for name in sorted(set(computed_components) | set(committed_components)):
        computed_component = computed_components.get(name, {})
        committed_component = committed_components.get(name, {})
        computed_digest = _digest_of(computed_component)
        recorded_digest = _digest_of(committed_component)
        if computed_digest == recorded_digest:
            continue
        drifts.append(
            ComponentDrift(
                component=name,
                tier=tier_of(name),
                files=_moved_files(computed_component, committed_component),
                recorded=recorded_digest,
                computed=computed_digest,
            )
        )
    return tuple(drifts)


def _components(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    components = record.get("components")
    if not isinstance(components, dict):
        return {}
    return {name: value for name, value in components.items() if isinstance(value, dict)}


def _digest_of(component: Mapping[str, Any]) -> str | None:
    digest = component.get("digest")
    return digest if isinstance(digest, str) else None


def _moved_files(computed: Mapping[str, Any], committed: Mapping[str, Any]) -> tuple[str, ...]:
    """The allowlisted files whose own digests differ — added, removed or edited."""
    computed_files = _files_of(computed)
    committed_files = _files_of(committed)
    names = set(computed_files) | set(committed_files)
    return tuple(
        sorted(rel for rel in names if computed_files.get(rel) != committed_files.get(rel))
    )


def _files_of(component: Mapping[str, Any]) -> dict[str, str | None]:
    files = component.get("files")
    if not isinstance(files, dict):
        return {}
    return {rel: digest if isinstance(digest, str) else None for rel, digest in files.items()}
