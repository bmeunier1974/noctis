"""The prompt ratchet — a prompt change cannot land undeclared.

:mod:`noctis.observability.prompt_id` computes what each LLM call site's prompt *is*; this module
is the check that the repo's committed statement of it — ``prompt_fingerprint.json`` at the repo
root — still matches, and that whatever moved arrived with a human explanation in
``docs/prompt-changelog.md``.

It is the engine ratchet's twin (:mod:`noctis.observability.engine_ratchet`) with one deliberate
difference: **there is no tier split**. Every prompt site is the same kind of thing, so every drift
is the same kind of event — a change to what a model is told, which must be declared. There is
nothing here that warns and passes; a hash whose explanation nobody wrote is exactly the silence
this ratchet exists to end.

**The declared-change rule**, which is the whole design, has two halves and needs both:

1. The newest changelog entry must **name the drifted site** (``## <date> — sites: author,
   ideation``). A nameless entry declares nothing: prose is not machine-readable, and a hash has to
   read back to a sentence about *that* site.
2. That entry must have **arrived after the committed record was written** — the record carries the
   digest of the entry that was newest when it was regenerated, and a matching digest means the
   changelog gained nothing. Otherwise yesterday's entry would be a standing permission to keep
   editing that site forever, which is a rubber stamp, not a ratchet.

**The precise rule**, in the order it is evaluated:

1. No readable committed record → **fail**. It is the baseline; without it nothing is checked.
2. A site's digest differs → **fail**, naming the site and the files that moved. When the drift is
   undeclared the message asks for a changelog entry; when it is declared the record simply was not
   regenerated yet, and the message says so. Either way the record is stale and the fix is named.
3. Otherwise → **ok**. Note that a changelog edit on its own never fails: nothing a model is told
   has moved, and failing there would push a contributor to regenerate — which would consume the
   entry they had just written, and consume it against no drift at all.

**And the rule ``--write`` adds** (:func:`regenerate`): regenerating is the fix this module
recommends in every message it prints, and it rewrites *every* site at once — so it cannot also be
the way an undeclared prompt change gets recorded, or the ratchet would only hold for contributors
who read the failure before typing the command it printed. ``--write`` therefore evaluates the
check first and, on **undeclared drift**, writes nothing and exits 1, printing the same
declare-or-restore guidance plus its refusal. Every other case still regenerates in one command —
declared drift, no drift at all, and a missing or unreadable record (there is nothing to compare
against, and that is how the baseline is created).

The comparison itself is pure — two records in, a structured result out — so every scenario is
testable by fingerprinting a temp tree, editing one prompt file, writing a changelog entry or not,
and fingerprinting it again. The I/O (reading the committed file, writing it, printing, the exit
code) lives in :func:`main`, which ``scripts/prompt_fingerprint.py`` and the pre-commit hook call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from noctis.observability.prompt_id import (
    SITE_ASSETS,
    default_root,
    file_digest,
    fingerprint,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Repo-root-relative, so the record sits beside the tree it describes — and beside
# ``engine_fingerprint.json``, its twin, which a reviewer sees move on its own clock.
RECORD_PATH = "prompt_fingerprint.json"

# The human half of the artifact: a dated entry per change, naming the sites it explains. It lives
# in ``docs/`` beside development.md, the page that explains this check to a contributor.
CHANGELOG_PATH = "docs/prompt-changelog.md"

# The one command that refreshes the record. Named in every message this module emits, and
# documented in docs/development.md — a check that does not say how to fix it gets disabled.
REGENERATE_COMMAND = "uv run python scripts/prompt_fingerprint.py --write"

RECORD_KIND = "noctis.prompt_fingerprint"

# How an entry names what it explains, quoted in the failure so nobody has to go looking.
_SITES_MARKER = "sites:"
_ENTRY_PREFIX = "## "
_FENCE = "```"

# The one thing ``--write`` will not do. Printed under the ordinary declare-or-restore guidance, in
# place of it, because "regenerate the record" is precisely the advice being refused here.
_WRITE_REFUSAL = (
    "refusing to regenerate: --write cannot be the way an undeclared prompt change gets recorded"
)

_DIGEST_CHARS = 16

Status = Literal["ok", "fail"]


@dataclass(frozen=True)
class ChangelogEntry:
    """One dated changelog entry: its heading, the sites it declares, and its identity.

    ``digest`` covers the whole entry text, so *amending* the newest entry — the natural way a
    second change in one PR gets declared — reads as a new declaration.
    """

    heading: str
    sites: tuple[str, ...]
    digest: str


@dataclass(frozen=True)
class SiteDrift:
    """One site whose prompt moved: the digests either side, the files, and whether declared."""

    site: str
    files: tuple[str, ...]
    recorded: str | None
    computed: str | None
    declared: bool

    def line(self) -> str:
        recorded = self.recorded or "null"
        computed = self.computed or "null"
        state = "declared" if self.declared else "UNDECLARED"
        return f"{self.site} ({state}): {recorded} -> {computed}"


@dataclass(frozen=True)
class RatchetResult:
    """The verdict: a status, why, and exactly which sites and files moved."""

    status: Status
    problems: tuple[str, ...] = ()
    drift: tuple[SiteDrift, ...] = ()
    changelog_heading: str | None = None

    @property
    def ok(self) -> bool:
        """Whether CI passes."""
        return self.status != "fail"

    @property
    def undeclared_drift(self) -> bool:
        """A site moved whose change the newest changelog entry does not declare.

        Read *off* the verdict, never folded into it: another reading of the same cases, not a
        third status, so ``--check`` reports the same stale record either way. Declared drift is a
        record that was simply not regenerated yet, and regenerating it is the fix.
        """
        return any(not moved.declared for moved in self.drift)

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
        lines = [f"{self.status.upper()}  prompt fingerprint ratchet ({RECORD_PATH})"]
        lines += [f"  {problem}" for problem in self.problems]
        for moved in self.drift:
            lines.append(f"  {moved.line()}")
            lines += [f"      {rel}" for rel in moved.files]
        if self.drift:
            # Which entry the check actually read, because "I wrote one and it still fails" is the
            # question this tool gets asked — a fenced or mis-headed entry is invisible otherwise.
            heading = self.changelog_heading or "none"
            lines.append(f"  newest {CHANGELOG_PATH} entry: {heading}")
        return lines


@dataclass(frozen=True)
class WriteOutcome:
    """What ``--write`` did: the record it wrote, or the verdict it refused to record.

    Exactly one of the two is meaningful — ``written`` is ``None`` on a refusal, and a refusal
    always carries the verdict it refused on, because the contributor needs to see *which* site
    moved to write the changelog entry that declares it.
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
    """The record for the tree at ``root``: per-site digests, per-file digests, changelog head.

    Per-file digests are what let a drift report name the file that moved instead of every file
    the site happens to cover — and, because a site digest is the digest of its files' digests,
    they also make the record self-verifying without the sources.
    """
    base = default_root() if root is None else Path(root)
    fp = fingerprint(base)
    sites: dict[str, Any] = {}
    for name, rel_paths in SITE_ASSETS.items():
        site = fp.sites[name]
        sites[name] = {
            "digest": site.digest,
            "note": site.note,
            "files": {
                rel: file_digest(base.joinpath(*rel.split("/"))) for rel in sorted(rel_paths)
            },
        }
    entry = newest_entry(_changelog_text(base))
    return {
        "kind": RECORD_KIND,
        "regenerate_with": REGENERATE_COMMAND,
        "changelog": {
            "path": CHANGELOG_PATH,
            "heading": None if entry is None else entry.heading,
            "digest": None if entry is None else entry.digest,
            "declares": [] if entry is None else list(entry.sites),
        },
        "sites": sites,
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
    if not isinstance(loaded, dict) or not isinstance(loaded.get("sites"), dict):
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
    """Regenerate the committed record — unless that would silently record an undeclared change.

    ``--write`` is the one-command fix every failure here names, and it rewrites *every* site at
    once. That is the loophole it has to close: a contributor is told to run this, and if it
    absorbed an undeclared prompt digest on the way, the ratchet's whole promise would come down to
    whether they read the failure before typing the command it printed. So the write path evaluates
    the check first and declines exactly the case the check exists to catch, leaving the tree
    checkable, and failing.

    Every other case still regenerates in one command: drift the changelog declares, no drift at
    all, and — the reason the guard cannot simply be "never write over drift" — a missing or
    unreadable committed record, which is how the baseline gets created in the first place.
    """
    base = default_root() if root is None else Path(root)
    verdict = check(base)
    if verdict.undeclared_drift:
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
    if committed is None:
        return RatchetResult(
            status="fail",
            problems=(
                f"no readable committed record at {RECORD_PATH} — it is the baseline this check "
                "compares against, so nothing can be verified without it",
            ),
            changelog_heading=_changelog_field(computed, "heading"),
        )

    declared = _declared_sites(computed, committed)
    drift = _drifts(computed, committed, declared)

    problems: list[str] = []
    undeclared = tuple(moved.site for moved in drift if not moved.declared)
    if undeclared:
        problems.append(
            f"undeclared prompt drift: {', '.join(undeclared)}. A prompt change must arrive with "
            f"its explanation — add a dated entry to the top of {CHANGELOG_PATH} whose heading "
            f'names the site(s), e.g. "## 2026-08-01 — {_SITES_MARKER} '
            f'{", ".join(undeclared)}" — or restore the wording'
        )
    remainder = tuple(moved.site for moved in drift if moved.declared)
    if remainder:
        problems.append(
            f"declared prompt drift: {', '.join(remainder)}. The changelog entry is there; the "
            "record was not regenerated with it"
        )

    return RatchetResult(
        status="fail" if drift else "ok",
        problems=tuple(problems),
        drift=drift,
        changelog_heading=_changelog_field(computed, "heading"),
    )


def newest_entry(text: str) -> ChangelogEntry | None:
    """The changelog's newest entry — the first ``## `` section — or ``None`` when there is none.

    Newest-first is how the file is written and how a human reads it, so "the newest entry" needs
    no dates parsed and no ordering trusted: it is the first one. The sites it declares are read
    from its **heading line** after ``sites:``, never from its prose — a declaration has to be
    somewhere a machine can find it, and a paragraph mentioning a module name is not that.

    Fenced code blocks are skipped, both when finding the entry and when finding where it ends:
    the page documents its own heading format in a fence, and a template is not a declaration.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    headings = _entry_headings(lines)
    if not headings:
        return None
    start = headings[0]
    end = next((index for index in headings if index > start), len(lines))
    heading = lines[start][len(_ENTRY_PREFIX) :].strip()
    body = "\n".join(lines[start:end]).strip()
    return ChangelogEntry(
        heading=heading,
        sites=_sites_in(heading),
        digest=hashlib.sha256(body.encode("utf-8")).hexdigest()[:_DIGEST_CHARS],
    )


def main(argv: Sequence[str] | None = None) -> int:
    """The entrypoint CI, pre-commit and contributors run. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="prompt_fingerprint",
        description=(
            "Check (or regenerate) the committed prompt-asset fingerprint record. Drift in any "
            "call site's prompt fails until the newest changelog entry names that site — and "
            "--write refuses to record a change the changelog does not declare: a prompt change "
            "must arrive with its explanation."
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
            f"Regenerate {RECORD_PATH} from the tree. Commit it in the same PR as the "
            f"{CHANGELOG_PATH} entry. Refuses (writing nothing, exit 1) on undeclared drift."
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


def _changelog_text(root: Path) -> str:
    path = root.joinpath(*CHANGELOG_PATH.split("/"))
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _entry_headings(lines: Sequence[str]) -> list[int]:
    """The line numbers that open an entry: ``## `` lines outside any code fence."""
    headings: list[int] = []
    in_fence = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith(_FENCE):
            in_fence = not in_fence
        elif not in_fence and line.startswith(_ENTRY_PREFIX):
            headings.append(index)
    return headings


def _sites_in(heading: str) -> tuple[str, ...]:
    """The site names a heading declares: everything after ``sites:``, comma-separated."""
    lowered = heading.lower()
    marker = lowered.find(_SITES_MARKER)
    if marker < 0:
        return ()
    listed = heading[marker + len(_SITES_MARKER) :]
    return tuple(name for name in (part.strip() for part in listed.split(",")) if name)


def _declared_sites(computed: Mapping[str, Any], committed: Mapping[str, Any]) -> frozenset[str]:
    """The sites the changelog declares *since* the committed record was written.

    Both halves of the rule live here: the newest entry has to name the site, **and** it has to be
    a different entry than the one the record was regenerated against. An entry that was already
    the head is a standing permission, not a declaration.
    """
    head = _changelog_field(computed, "digest")
    if head is None or head == _changelog_field(committed, "digest"):
        return frozenset()
    declares = _changelog_block(computed).get("declares")
    if not isinstance(declares, list):
        return frozenset()
    return frozenset(name for name in declares if isinstance(name, str))


def _changelog_block(record: Mapping[str, Any]) -> Mapping[str, Any]:
    block = record.get("changelog")
    return block if isinstance(block, dict) else {}


def _changelog_field(record: Mapping[str, Any], field: str) -> str | None:
    value = _changelog_block(record).get(field)
    return value if isinstance(value, str) else None


def _drifts(
    computed: Mapping[str, Any], committed: Mapping[str, Any], declared: frozenset[str]
) -> tuple[SiteDrift, ...]:
    """Every site whose digest differs, with the files that moved under it.

    Two nulls are not drift (neither side could identify the site, so nothing is known to have
    moved) — the same rule :func:`noctis.observability.prompt_id.compare` takes.
    """
    computed_sites = _sites(computed)
    committed_sites = _sites(committed)
    drifts: list[SiteDrift] = []
    for name in sorted(set(computed_sites) | set(committed_sites)):
        computed_site = computed_sites.get(name, {})
        committed_site = committed_sites.get(name, {})
        computed_digest = _digest_of(computed_site)
        recorded_digest = _digest_of(committed_site)
        if computed_digest == recorded_digest:
            continue
        drifts.append(
            SiteDrift(
                site=name,
                files=_moved_files(computed_site, committed_site),
                recorded=recorded_digest,
                computed=computed_digest,
                declared=name in declared,
            )
        )
    return tuple(drifts)


def _sites(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    sites = record.get("sites")
    if not isinstance(sites, dict):
        return {}
    return {name: value for name, value in sites.items() if isinstance(value, dict)}


def _digest_of(site: Mapping[str, Any]) -> str | None:
    digest = site.get("digest")
    return digest if isinstance(digest, str) else None


def _moved_files(computed: Mapping[str, Any], committed: Mapping[str, Any]) -> tuple[str, ...]:
    """The allowlisted files whose own digests differ — added, removed or edited."""
    computed_files = _files_of(computed)
    committed_files = _files_of(committed)
    names = set(computed_files) | set(committed_files)
    return tuple(
        sorted(rel for rel in names if computed_files.get(rel) != committed_files.get(rel))
    )


def _files_of(site: Mapping[str, Any]) -> dict[str, str | None]:
    files = site.get("files")
    if not isinstance(files, dict):
        return {}
    return {rel: digest if isinstance(digest, str) else None for rel, digest in files.items()}
