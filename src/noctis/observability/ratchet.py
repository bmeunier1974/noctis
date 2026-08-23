"""The ratchet mechanics: a committed record, a check against the tree, and a guarded ``--write``.

A ratchet is a *record* plus a *rule*. Noctis has two of them — the engine fingerprint
(:mod:`noctis.observability.engine_ratchet`) and the prompt fingerprint
(:mod:`noctis.observability.prompt_ratchet`) — on separate clocks, in separate files, with
separate commands. Everything except the rule is the same mechanism, and it is written here once:
a policy is a :class:`RatchetSpec` (what the record is called, what it holds, how the tree is
identified) plus one :class:`Judgement`-returning callable that *is* its rule. A mechanics fix can
then no longer land in one ratchet and be missed in the other.

**The contract, in four cases**, identical for every policy:

1. **No readable committed record** → fail. It is the baseline; without it nothing is checked.
   This one is decided here, not by the policy: there is nothing yet to have an opinion about.
2. **Entries moved** → the policy decides. :func:`compare_records` hands the judge the two records
   and the moves, and the judge returns the status, the problems, the tagged drifts in the order
   they print, and whether ``--write`` may record them.
3. **Nothing moved** → ok, and the report says the committed record matches this tree.
4. ``--write`` (:func:`regenerate`) **evaluates the check first** and writes nothing when the
   policy marked the verdict :attr:`~Judgement.refuse_write`. Regenerating is the fix every
   message here recommends and it rewrites *every* entry at once, so it cannot also be the way an
   undeclared change gets recorded — or a ratchet's whole promise would come down to whether the
   contributor read the failure before typing the command it printed.

**What moved is one rule**, :func:`noctis.observability.engine_id.compare`, over the two records
projected to their ``name -> digest`` maps: an entry present on one side only moved (a fingerprint
surface *appearing* is exactly the news a ratchet exists to publish, and silence on it is the
failure), two nulls did not, and a non-string value in a hand-editable JSON record reads as null.
:func:`_moved` attaches the files whose own digests differ, so a report names the file that moved
rather than every file the entry covers.

The comparison is pure — two records in, a structured result out — so every scenario is testable
by fingerprinting a temp tree, editing one file, and fingerprinting it again. The I/O (reading the
committed file, writing it, printing, the exit code) lives in :func:`main`, which each policy's
script and pre-commit hook call through a one-line binding.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from noctis.observability.engine_id import compare, default_root, file_digest

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

Status = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class Moved:
    """One entry whose digest moved, before any policy has said what that means.

    The mechanics can see *that* something moved and *which files* did; whether it is a warning,
    a failure or a declared change is the policy's word, added by :meth:`tagged`.
    """

    name: str
    files: tuple[str, ...]
    recorded: str | None
    computed: str | None

    def tagged(self, tag: str) -> Drift:
        """This move, filed under the one word the policy's report prints beside it."""
        return Drift(
            name=self.name,
            files=self.files,
            recorded=self.recorded,
            computed=self.computed,
            tag=tag,
        )


@dataclass(frozen=True)
class Drift(Moved):
    """A move a policy has judged: its one word, the digests either side, the files that moved."""

    tag: str

    def line(self) -> str:
        """``name (tag): recorded -> computed`` — the line a report prints for one drift."""
        recorded = self.recorded or "null"
        computed = self.computed or "null"
        return f"{self.name} ({self.tag}): {recorded} -> {computed}"


@dataclass(frozen=True)
class Judgement:
    """What a policy says about a list of moves: the verdict, why, and what ``--write`` may do."""

    status: Status
    problems: tuple[str, ...] = ()
    # Tagged, and in the order the report prints them — a policy that ranks its tiers orders here.
    drifts: tuple[Drift, ...] = ()
    # Read *off* the verdict, never folded into it: another reading of the same cases, not an
    # extra status, so ``--check`` reports exactly what it always did.
    refuse_write: bool = False
    # Lines printed after the drifts, e.g. which changelog entry the check actually read.
    footer: tuple[str, ...] = ()


def _no_footer(record: Mapping[str, Any]) -> tuple[str, ...]:
    """A policy whose report ends at the drift lines. The engine ratchet is one."""
    return ()


@dataclass(frozen=True)
class RatchetSpec:
    """One ratchet: what its record is, how the tree is identified, and the rule over the two.

    Data plus one judging callable. Everything a policy needs to say about *itself* is a field
    here; everything it needs to say about a comparison is :attr:`judge`.
    """

    title: str  # "engine fingerprint ratchet" — the first word of every report line but the first
    record_path: str  # repo-root-relative, so the record sits beside the tree it describes
    record_kind: str
    regenerate_command: str  # the one command every message names, or the check gets disabled
    entries_key: str  # the record key holding the entries, and the one :func:`load_record` demands
    asset_paths: Mapping[str, tuple[str, ...]]  # name -> the allowlisted files under it
    identity: Callable[[Path], Mapping[str, tuple[str | None, str | None]]]  # -> (digest, note)
    header: Callable[[Path], dict[str, Any]]  # the policy's own top-level record fields
    judge: Callable[[Mapping[str, Any], Mapping[str, Any], tuple[Moved, ...]], Judgement]
    write_refusal: str  # printed in place of the regenerate advice when ``--write`` declines
    prog: str
    description: str
    write_help: str
    # The footer of the one verdict no judge sees: a record that is missing or unreadable.
    footer: Callable[[Mapping[str, Any]], tuple[str, ...]] = _no_footer


@dataclass(frozen=True)
class RatchetResult:
    """The verdict: a status, why, and exactly which entries and files moved."""

    spec: RatchetSpec
    status: Status
    problems: tuple[str, ...] = ()
    drifts: tuple[Drift, ...] = ()
    refuse_write: bool = False
    footer: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the check passes. A warning is visible, not blocking."""
        return self.status != "fail"

    def report(self) -> str:
        """The human-readable verdict CI prints: what moved, in which files, and the fix."""
        if self.status == "ok":
            advice = "the committed record matches this tree"
        else:
            advice = f"regenerate the record: {self.spec.regenerate_command}"
        return "\n".join([*self.findings(), f"  {advice}"])

    def findings(self) -> list[str]:
        """The header, the problems and the files that moved — everything but the closing advice.

        Split out because :meth:`WriteOutcome.report` prints the same findings and then
        *contradicts* that closing advice ("regenerate the record" is exactly what it is refusing),
        and one renderer of a verdict is better than two that drift apart.
        """
        lines = [f"{self.status.upper()}  {self.spec.title} ({self.spec.record_path})"]
        lines += [f"  {problem}" for problem in self.problems]
        for drift in self.drifts:
            lines.append(f"  {drift.line()}")
            lines += [f"      {rel}" for rel in drift.files]
        lines += [f"  {line}" for line in self.footer]
        return lines


@dataclass(frozen=True)
class WriteOutcome:
    """What ``--write`` did: the record it wrote, or the verdict it refused to record.

    Exactly one of the two is meaningful — ``written`` is ``None`` on a refusal, and a refusal
    always carries the verdict it refused on, because the contributor needs to see *which* entry
    moved to decide what to do about it.
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
            return f"wrote {self.verdict.spec.record_path} ({self.written})"
        return "\n".join([*self.verdict.findings(), f"  {self.verdict.spec.write_refusal}"])


def build_record(spec: RatchetSpec, root: Path | None = None) -> dict[str, Any]:
    """The record for the tree at ``root``: the policy's header, then digests and per-file digests.

    Per-file digests are what let a drift report name the file that moved instead of every file
    the entry happens to cover.
    """
    base = _base(root)
    identity = spec.identity(base)
    entries: dict[str, Any] = {}
    for name, rel_paths in spec.asset_paths.items():
        digest, note = identity.get(name, (None, None))
        entries[name] = {
            "digest": digest,
            "note": note,
            "files": {
                rel: file_digest(base.joinpath(*rel.split("/"))) for rel in sorted(rel_paths)
            },
        }
    return {
        "kind": spec.record_kind,
        "regenerate_with": spec.regenerate_command,
        **spec.header(base),
        spec.entries_key: entries,
    }


def load_record(spec: RatchetSpec, root: Path | None = None) -> dict[str, Any] | None:
    """The committed record, or ``None`` when it is absent or unreadable as a record."""
    path = _record_path(spec, root)
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(loaded, dict) or not isinstance(loaded.get(spec.entries_key), dict):
        return None
    return loaded


def write_record(spec: RatchetSpec, root: Path | None = None) -> Path:
    """Write the record for ``root``, unconditionally. Sorted keys, so its diff reads cleanly.

    The plain writer: it states the tree as it is and asks no questions, which is what creating a
    baseline in a fresh tree needs. The *policy* about when regenerating is allowed lives one level
    up in :func:`regenerate`, so the write path can refuse without this function — or
    :func:`build_record` — ever having to know about the previous record.
    """
    path = _record_path(spec, root)
    record = build_record(spec, root)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def regenerate(spec: RatchetSpec, root: Path | None = None) -> WriteOutcome:
    """Regenerate the committed record — unless the policy refuses to have it recorded this way.

    Case 4 of the contract: evaluate the check first, and decline exactly what the judge marked
    :attr:`~Judgement.refuse_write`, leaving the tree checkable, and failing. Every other case
    still regenerates in one command — including a missing or unreadable committed record, which
    is the reason the guard cannot simply be "never write over drift": it is how the baseline gets
    created in the first place.
    """
    base = _base(root)
    verdict = check(spec, base)
    if verdict.refuse_write:
        return WriteOutcome(verdict=verdict, written=None)
    return WriteOutcome(verdict=verdict, written=write_record(spec, base))


def check(spec: RatchetSpec, root: Path | None = None) -> RatchetResult:
    """Recompute the tree's record and compare it with the committed one."""
    base = _base(root)
    return compare_records(spec, build_record(spec, base), load_record(spec, base))


def compare_records(
    spec: RatchetSpec, computed: Mapping[str, Any], committed: Mapping[str, Any] | None
) -> RatchetResult:
    """Compare a freshly computed record with the committed one. Pure: no files, no clock.

    The missing baseline is decided here (case 1); everything else is the policy's word on the
    moves :func:`_moved` found.
    """
    if committed is None:
        return RatchetResult(
            spec=spec,
            status="fail",
            problems=(
                f"no readable committed record at {spec.record_path} — it is the baseline this "
                "check compares against, so nothing can be verified without it",
            ),
            footer=spec.footer(computed),
        )
    verdict = spec.judge(computed, committed, _moved(spec, computed, committed))
    return RatchetResult(
        spec=spec,
        status=verdict.status,
        problems=verdict.problems,
        drifts=verdict.drifts,
        refuse_write=verdict.refuse_write,
        footer=verdict.footer,
    )


def main(spec: RatchetSpec, argv: Sequence[str] | None = None) -> int:
    """The entrypoint CI, pre-commit and contributors run. Returns the process exit code."""
    parser = argparse.ArgumentParser(prog=spec.prog, description=spec.description)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--check",
        action="store_true",
        help="Compare the tree with the committed record (the default action).",
    )
    action.add_argument("--write", action="store_true", help=spec.write_help)
    parser.add_argument(
        "--root",
        default=None,
        help="Repo root to check (defaults to this checkout).",
    )
    args = parser.parse_args(argv)
    root = None if args.root is None else Path(args.root)

    if args.write:
        outcome = regenerate(spec, root)
        print(outcome.report())
        return 0 if outcome.ok else 1

    result = check(spec, root)
    print(result.report())
    return 0 if result.ok else 1


def _base(root: Path | None) -> Path:
    return default_root() if root is None else Path(root)


def _record_path(spec: RatchetSpec, root: Path | None) -> Path:
    return _base(root) / spec.record_path


def _moved(
    spec: RatchetSpec, computed: Mapping[str, Any], committed: Mapping[str, Any]
) -> tuple[Moved, ...]:
    """Every entry whose digest moved, with the files that moved under it, name-sorted.

    Which names moved is :func:`noctis.observability.engine_id.compare` over the two records'
    digest maps and nothing else — so an entry present on one side only is drift, two nulls are
    not, and a drift report can never disagree with a resume policy about the same edit.
    """
    computed_entries = _entries(spec, computed)
    committed_entries = _entries(spec, committed)
    return tuple(
        Moved(
            name=name,
            files=_moved_files(computed_entries.get(name, {}), committed_entries.get(name, {})),
            recorded=_digest_of(committed_entries.get(name, {})),
            computed=_digest_of(computed_entries.get(name, {})),
        )
        for name in sorted(compare(_digests(committed_entries), _digests(computed_entries)))
    )


def _entries(spec: RatchetSpec, record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """The record's entries, tolerant of a hand-edit: a name whose value is not an entry is kept,
    reading as an entry nothing is known about, because the *name* is what drift is computed on."""
    entries = record.get(spec.entries_key)
    if not isinstance(entries, dict):
        return {}
    return {name: value if isinstance(value, dict) else {} for name, value in entries.items()}


def _digests(entries: Mapping[str, Mapping[str, Any]]) -> dict[str, str | None]:
    return {name: _digest_of(entry) for name, entry in entries.items()}


def _digest_of(entry: Mapping[str, Any]) -> str | None:
    digest = entry.get("digest")
    return digest if isinstance(digest, str) else None


def _moved_files(computed: Mapping[str, Any], committed: Mapping[str, Any]) -> tuple[str, ...]:
    """The allowlisted files whose own digests differ — added, removed or edited."""
    computed_files = _files_of(computed)
    committed_files = _files_of(committed)
    names = set(computed_files) | set(committed_files)
    return tuple(
        sorted(rel for rel in names if computed_files.get(rel) != committed_files.get(rel))
    )


def _files_of(entry: Mapping[str, Any]) -> dict[str, str | None]:
    files = entry.get("files")
    if not isinstance(files, dict):
        return {}
    return {rel: digest if isinstance(digest, str) else None for rel, digest in files.items()}
