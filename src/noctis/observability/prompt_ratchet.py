"""The prompt ratchet — a prompt change cannot land undeclared.

:mod:`noctis.observability.prompt_id` computes what each LLM call site's prompt *is*; this module
is the check that the repo's committed statement of it — ``prompt_fingerprint.json`` at the repo
root — still matches, and that whatever moved arrived with a human explanation in
``docs/prompt-changelog.md``.

This file holds the **rule** and nothing else: the record, the comparison, ``--write`` and the
report are the shared mechanics in :mod:`noctis.observability.ratchet`, which every ratchet runs
on, and the rule below is the :class:`~noctis.observability.ratchet.RatchetSpec` plus one judging
function it is given — with the changelog reader the rule needs, which is policy and lives here.
What moved is decided by :func:`~noctis.observability.engine_id.compare`, the one null rule.

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

Every report that names a drift also names the changelog entry the check *read*, because "I wrote
one and it still fails" is the question this tool gets asked — and so does the report for a missing
record, which is the one verdict no judge sees.

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
code) lives in :func:`main`, which ``scripts/prompt_fingerprint.py`` and the pre-commit hook call
through a one-line binding.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from noctis.observability import ratchet
from noctis.observability.prompt_id import SITE_ASSETS, fingerprint
from noctis.observability.ratchet import (
    Judgement,
    Moved,
    RatchetResult,
    RatchetSpec,
    WriteOutcome,
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


@dataclass(frozen=True)
class ChangelogEntry:
    """One dated changelog entry: its heading, the sites it declares, and its identity.

    ``digest`` covers the whole entry text, so *amending* the newest entry — the natural way a
    second change in one PR gets declared — reads as a new declaration.
    """

    heading: str
    sites: tuple[str, ...]
    digest: str


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


def _changelog_text(root: Path) -> str:
    path = root.joinpath(*CHANGELOG_PATH.split("/"))
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _identity(root: Path) -> dict[str, tuple[str | None, str | None]]:
    """Each site's digest and why it is null, from the one module that computes them."""
    computed = fingerprint(root)
    return {name: (site.digest, site.note) for name, site in computed.sites.items()}


def _header(root: Path) -> dict[str, Any]:
    """The changelog head this record is written against — the second half of the rule.

    Storing the entry's digest is what makes "arrived *after* the record" checkable at all: a
    matching digest means the changelog gained nothing since, so the entry is yesterday's standing
    permission rather than today's declaration.
    """
    entry = newest_entry(_changelog_text(root))
    return {
        "changelog": {
            "path": CHANGELOG_PATH,
            "heading": None if entry is None else entry.heading,
            "digest": None if entry is None else entry.digest,
            "declares": [] if entry is None else list(entry.sites),
        }
    }


def _footer(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Which entry the check actually read, from the record it computed for this tree.

    "I wrote one and it still fails" is the question this tool gets asked, and a fenced or
    mis-headed entry is invisible without this line — so it is printed for a missing baseline too,
    where there is no drift yet to hang it off.
    """
    heading = _changelog_field(record, "heading") or "none"
    return (f"newest {CHANGELOG_PATH} entry: {heading}",)


def _judge(
    computed: Mapping[str, Any], committed: Mapping[str, Any], moved: tuple[Moved, ...]
) -> Judgement:
    """The declared-change rule — the module docstring, as code."""
    declared = _declared_sites(computed, committed)
    drifts = tuple(
        move.tagged("declared" if move.name in declared else "UNDECLARED") for move in moved
    )
    undeclared = tuple(drift.name for drift in drifts if drift.tag == "UNDECLARED")
    remainder = tuple(drift.name for drift in drifts if drift.tag == "declared")

    problems: list[str] = []
    if undeclared:
        problems.append(
            f"undeclared prompt drift: {', '.join(undeclared)}. A prompt change must arrive with "
            f"its explanation — add a dated entry to the top of {CHANGELOG_PATH} whose heading "
            f'names the site(s), e.g. "## 2026-08-01 — {_SITES_MARKER} '
            f'{", ".join(undeclared)}" — or restore the wording'
        )
    if remainder:
        problems.append(
            f"declared prompt drift: {', '.join(remainder)}. The changelog entry is there; the "
            "record was not regenerated with it"
        )

    return Judgement(
        status="fail" if drifts else "ok",
        problems=tuple(problems),
        drifts=drifts,
        # Declared drift is a record that was simply not regenerated yet, and regenerating it is
        # the fix — so only an undeclared site is what ``--write`` will not launder.
        refuse_write=bool(undeclared),
        footer=_footer(computed) if drifts else (),
    )


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


SPEC = RatchetSpec(
    title="prompt fingerprint ratchet",
    record_path=RECORD_PATH,
    record_kind=RECORD_KIND,
    regenerate_command=REGENERATE_COMMAND,
    entries_key="sites",
    asset_paths=SITE_ASSETS,
    identity=_identity,
    header=_header,
    judge=_judge,
    write_refusal=_WRITE_REFUSAL,
    prog="prompt_fingerprint",
    description=(
        "Check (or regenerate) the committed prompt-asset fingerprint record. Drift in any "
        "call site's prompt fails until the newest changelog entry names that site — and "
        "--write refuses to record a change the changelog does not declare: a prompt change "
        "must arrive with its explanation."
    ),
    write_help=(
        f"Regenerate {RECORD_PATH} from the tree. Commit it in the same PR as the "
        f"{CHANGELOG_PATH} entry. Refuses (writing nothing, exit 1) on undeclared drift."
    ),
    footer=_footer,
)


def build_record(root: Path | None = None) -> dict[str, Any]:
    """The record for the tree at ``root``: per-site digests, per-file digests, changelog head."""
    return ratchet.build_record(SPEC, root)


def load_record(root: Path | None = None) -> dict[str, Any] | None:
    """The committed record, or ``None`` when it is absent or unreadable as a record."""
    return ratchet.load_record(SPEC, root)


def write_record(root: Path | None = None) -> Path:
    """Write the record for ``root``, unconditionally — the plain writer a baseline needs."""
    return ratchet.write_record(SPEC, root)


def regenerate(root: Path | None = None) -> WriteOutcome:
    """Regenerate the committed record — unless that would record an undeclared prompt change."""
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
