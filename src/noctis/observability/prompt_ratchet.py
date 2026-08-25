"""The prompt ratchet — a prompt change cannot land undeclared.

:mod:`noctis.observability.prompt_id` computes what each LLM call site's prompt *is*; this module
is the check that the repo's committed statement of it — ``prompt_fingerprint.json`` at the repo
root — still matches, and that whatever moved arrived with a human explanation in
``docs/prompt-changelog.md``.

This file holds the **rule** and nothing else: the record, the comparison, ``--write`` and the
report are the shared mechanics in :mod:`noctis.observability.ratchet`, which every ratchet runs
on, and the rule below is the :class:`~noctis.observability.ratchet.RatchetSpec` plus one judging
function it is given. How a changelog entry is *read* is shared too
(:mod:`noctis.observability.changelog`: the newest entry, its digest, its heading's clauses); what
this policy adds is the one clause it binds — ``sites:`` — and what naming a site there permits.
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

from pathlib import Path
from typing import TYPE_CHECKING, Any

from noctis.observability import changelog, ratchet
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

# The one changelog clause this policy binds: how an entry names what it explains, quoted in the
# failure so nobody has to go looking. Every other clause on a heading is somebody else's.
_SITES_CLAUSE = "sites"

# The one thing ``--write`` will not do. Printed under the ordinary declare-or-restore guidance, in
# place of it, because "regenerate the record" is precisely the advice being refused here.
_WRITE_REFUSAL = (
    "refusing to regenerate: --write cannot be the way an undeclared prompt change gets recorded"
)


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
    entry = changelog.read_entry(root, CHANGELOG_PATH)
    return changelog.header(CHANGELOG_PATH, entry, _sites_named_by(entry))


def _sites_named_by(entry: changelog.ChangelogEntry | None) -> tuple[str, ...]:
    """The sites one entry declares: the names on its ``sites:`` clause, and nothing else.

    The whole of what this policy binds of the shared grammar — a heading's other clauses, and its
    prose, declare nothing here.
    """
    return () if entry is None else entry.names(_SITES_CLAUSE)


def _footer(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Which entry the check actually read, from the record it computed for this tree.

    "I wrote one and it still fails" is the question this tool gets asked, and a fenced or
    mis-headed entry is invisible without this line — so it is printed for a missing baseline too,
    where there is no drift yet to hang it off.
    """
    return changelog.footer(CHANGELOG_PATH, record)


def _judge(
    computed: Mapping[str, Any], committed: Mapping[str, Any], moved: tuple[Moved, ...]
) -> Judgement:
    """The declared-change rule — the module docstring, as code."""
    declared = changelog.declared_since(computed, committed)
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
            f'names the site(s), e.g. "## 2026-08-01 — {_SITES_CLAUSE}: '
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
