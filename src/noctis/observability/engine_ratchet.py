"""The CI ratchet — behavioural drift cannot land silently.

:mod:`noctis.observability.engine_id` computes what the engine *is*; this module is the check
that the repo's committed statement of it — ``engine_fingerprint.json`` at the repo root — still
matches, and that a change to the **arbiter** came with a declared version bump.

**The tier split, which is the whole design.** The line is the one
:data:`~noctis.observability.engine_id.ARBITER_COMPONENTS` already draws — read from there, never
restated here, because two copies of that set would eventually disagree and the disagreement
would be silent.

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

The comparison itself is pure — two records in, a structured result out — so every scenario is
testable by fingerprinting a temp tree, editing one file, and fingerprinting it again. The I/O
(reading the committed file, writing it, printing, the exit code) lives in :func:`main`, which
``scripts/engine_fingerprint.py`` and the pre-commit hook call.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from noctis.observability.engine_id import (
    ARBITER_COMPONENTS,
    COMPONENT_PATHS,
    default_root,
    file_digest,
    fingerprint,
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

Status = Literal["ok", "warn", "fail"]
Tier = Literal["arbiter", "searcher"]


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

    def report(self) -> str:
        """The human-readable verdict CI prints: what moved, in which files, and the fix."""
        lines = [f"{self.status.upper()}  engine fingerprint ratchet ({RECORD_PATH})"]
        lines += [f"  {problem}" for problem in self.problems]
        for drift in (*self.arbiter_drift, *self.searcher_drift):
            lines.append(f"  {drift.line()}")
            lines += [f"      {rel}" for rel in drift.files]
        if self.status == "ok":
            lines.append("  the committed record matches this tree")
        else:
            lines.append(f"  regenerate the record: {REGENERATE_COMMAND}")
        return "\n".join(lines)


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
    """Regenerate the committed record for ``root``. Sorted keys, so its diff reads cleanly."""
    path = _record_path(root)
    record = build_record(root)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


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
            "passes, naming the component and the files that moved."
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
        help=f"Regenerate {RECORD_PATH} from the tree. Commit it in the same PR.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Repo root to check (defaults to this checkout).",
    )
    args = parser.parse_args(argv)
    root = None if args.root is None else Path(args.root)

    if args.write:
        path = write_record(root)
        print(f"wrote {RECORD_PATH} ({path})")
        return 0

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
                tier="arbiter" if name in ARBITER_COMPONENTS else "searcher",
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
