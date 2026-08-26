"""The CI ratchet — behavioural drift cannot land silently.

:mod:`noctis.observability.engine_id` computes what the engine *is*; this module is the check
that the repo's committed statement of it — ``engine_fingerprint.json`` at the repo root — still
matches, and that a change to the **arbiter** arrived *declared*.

This file holds the **rule** and nothing else: the record, the comparison, ``--write`` and the
report are the shared mechanics in :mod:`noctis.observability.ratchet`, which every ratchet runs
on, and the rule below is the :class:`~noctis.observability.ratchet.RatchetSpec` plus one judging
function it is given. How a changelog entry is *read* is shared too
(:mod:`noctis.observability.changelog`: the newest entry, its digest, its heading's clauses); what
this policy adds is the two clauses it binds — ``components:`` and ``behaviour: unchanged`` — and
what naming a component there permits. What moved is decided by
:func:`~noctis.observability.engine_id.compare` — the one null rule — so this check and the resume
policy can never disagree about the same edit.

**The tier split, which is the whole design.** The line is the one
:data:`~noctis.observability.engine_id.ARBITER_COMPONENTS` already draws, and it is read through
:func:`~noctis.observability.engine_id.tier_of` — the single classifier this check and the resume
policy (:mod:`noctis.observability.engine_change`) both call. Never restated here: two copies of
that set would eventually disagree, and the disagreement would be silent.

* **Arbiter tier** (the components that decide what passes and what a number means): drift
  **fails**. This is the one change that invalidates every stored champion comparison, so it can
  never land undeclared.
* **Searcher tier** (how candidates are found, what the model is told, the shipped profiles, the
  seed library, the memory seed, the record schema): drift **warns and passes**, naming the
  component and the files. Improving the searcher must not invalidate an experiment whose
  arbiter held still — and a ratchet that fires on a docstring edit gets disabled, which is
  worse than no ratchet at all.

**The two ways an arbiter move is declared**, because a digest is a content hash and a version is
a claim about results:

1. **An ``ENGINE_VERSION`` bump** — "these numbers are no longer comparable". The declaration for
   a behaviour change, and the only one that moves the key runs are compared on.
2. **A no-op entry** at the top of ``docs/engine-changelog.md`` — "this edit was mechanical".
   A rename, an import path, a docstring, a type annotation, a deleted pass-through: the digest
   moves on every byte, while ``ENGINE_VERSION`` means incomparable, so bumping for one of those
   would assert a false incomparability. The entry's heading carries **both** clauses,
   ``components: <name>[, <name>…]`` and ``behaviour: unchanged``; ``components:`` alone narrates
   (the page is the arbiter's human history, and it may describe a bump), and no entry is ever
   demanded for a bump. What the declaration lifts is **only** the ``--write`` refusal — never the
   failure, and never anything the digest's other readers do (the resume policy and
   ``comparable_key`` keep partitioning on it, so a wrong claim costs a contributor a false
   refusal and can never pool two engines' numbers).

**The precise rule**, in the order it is evaluated:

1. No readable committed record → **fail**. It is the baseline; without it nothing is checked.
2. The record declares another ``ENGINE_VERSION`` than this tree → **fail** as stale. A bump is
   only half the deal: the record has to be regenerated in the same PR so the diff shows a
   reviewer exactly which behavioural surface moved.
3. An arbiter component's digest differs and the versions **differ** → **fail**: the bump is
   there, the record simply was not regenerated. A no-op entry beside a bump is not read as a
   contradiction — both may have happened in one PR — so the declaration decides nothing here.
4. An arbiter component's digest differs and the versions **agree** → **fail** either way, split
   on the declaration: an **undeclared** component names all three outs (bump, declare a no-op,
   restore the behaviour) and is what ``--write`` refuses; a **declared no-op** is a record that
   was simply not regenerated with its entry, exactly as a declared bump is.
5. Only searcher components differ → **warn**, exit zero. The record is stale in the honest,
   tolerated way; the warning names what to regenerate, and blocks nothing.

So "a stale record is what the contributor is told to fix" and "searcher drift warns and passes"
compose: staleness is always *reported* and always names the regeneration command; whether it is
*blocking* is decided by which tier moved.

**The declared-change rule has two halves and needs both** (the prompt ratchet's, verbatim): the
newest entry must **name the component**, and it must have **arrived after the committed record
was written** — the record carries the digest of the entry that was newest when it was
regenerated, and a matching digest means the page gained nothing. Otherwise yesterday's no-op
would be a standing permission to keep editing that component forever. A missing page, an
entry-less one, an entry naming another component and an entry naming a searcher component
(nothing there is blocked, so there is nothing to lift) all declare nothing.

**And the rule ``--write`` adds**: regenerating is the fix this module recommends in every message
it prints, and it rewrites *every* component at once — so it cannot also be the way case 4's
undeclared arbiter move gets recorded, or the ratchet would only hold for contributors who read
the failure before typing the command it printed. ``--write`` therefore evaluates the check first
and, on **undeclared arbiter drift with the versions in agreement**, writes nothing and exits 1,
printing the same three outs plus its refusal. Every other case still regenerates in one command —
searcher-only drift, an arbiter move whose bump is declared, an arbiter move declared a no-op, no
drift at all, and a missing or unreadable record (there is nothing to compare against, and that is
how the baseline is created). An arbiter move must arrive *declared*; the tool will not launder
one.

Every report that names arbiter drift also names the changelog entry the check *read*, because "I
wrote one and it still fails" is the question this tool gets asked — and so does the report for a
missing record, which is the one verdict no judge sees. Searcher-only drift never does: there is
nothing to declare, and a check that always says something is one people skip.

The comparison itself is pure — two records in, a structured result out — so every scenario is
testable by fingerprinting a temp tree, editing one file, and fingerprinting it again. The I/O
(reading the committed file, writing it, printing, the exit code) lives in :func:`main`, which
``scripts/engine_fingerprint.py`` and the pre-commit hook call, and the refusal is a decision in
that write path rather than a change to what a comparison *means*.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from noctis.observability import changelog, ratchet
from noctis.observability.engine_id import (
    COMPONENT_PATHS,
    ENGINE_VERSION,
    Tier,
    fingerprint,
    tier_of,
)
from noctis.observability.ratchet import (
    Drift,
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

# The human half of the artifact: a dated entry per *mechanical* arbiter move, naming the
# components it explains. It lives in ``docs/`` beside development.md, the page that explains this
# check to a contributor — and beside ``prompt-changelog.md``, its twin on the other clock.
CHANGELOG_PATH = "docs/engine-changelog.md"

# The two changelog clauses this policy binds, quoted in the failure so nobody has to go looking:
# the components an entry explains, and the marker that says the edit changed no behaviour. Both
# are required — ``components:`` on its own narrates. Every other clause on a heading is somebody
# else's, and the value below is matched exactly (the key, by the shared reader, is not).
_COMPONENTS_CLAUSE = "components"
_BEHAVIOUR_CLAUSE = "behaviour"
_NO_OP_VALUE = "unchanged"

# The word a declared no-op is filed under in a report, appended to the tier the one classifier
# gave it: the tier word stays the policy's, and one line says which of the two arbiter cases
# this is.
_NO_OP_TAG = "declared no-op"

# Where the declared version lives, quoted in the failure so nobody has to go looking.
_VERSION_SOURCE = "src/noctis/observability/engine_id.py"

# The one thing ``--write`` will not do. Printed under the ordinary three-outs guidance, in place
# of it, because "regenerate the record" is precisely the advice being refused here.
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
    """The declared comparison key and the changelog head — what makes a record a *statement*.

    The version is **declared** by this checkout's constant, never read out of the tree being
    stated, which is exactly why a record can be stale. The changelog head is the other half of
    the declared-no-op rule: storing the entry's digest is what makes "arrived *after* the record"
    checkable at all, and a matching digest means the page gained nothing since.
    """
    entry = changelog.read_entry(root, CHANGELOG_PATH)
    return {
        "engine_version": ENGINE_VERSION,
        **changelog.header(CHANGELOG_PATH, entry, _no_op_components(entry)),
    }


def _no_op_components(entry: changelog.ChangelogEntry | None) -> tuple[str, ...]:
    """The components one entry declares mechanical: both clauses together, or nothing.

    The whole of what this policy binds of the shared grammar. ``components:`` on its own
    narrates — the page is the arbiter's human history and an entry may describe a bump — so only
    the ``behaviour: unchanged`` marker beside it declares anything to this check.
    """
    if entry is None or entry.names(_BEHAVIOUR_CLAUSE) != (_NO_OP_VALUE,):
        return ()
    return entry.names(_COMPONENTS_CLAUSE)


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
    """The tier rule, the version agreement and the declared no-op — the docstring, as code."""
    computed_version = _version_of(computed)
    recorded_version = _version_of(committed)
    agreed = recorded_version == computed_version
    # The declaration is read only where it can decide anything: with the versions in
    # disagreement the record is stale whatever the page says, and case 3 already names the fix.
    declared = changelog.declared_since(computed, committed) if agreed else frozenset()
    drifts = tuple(move.tagged(_tag(move.name, declared)) for move in moved)
    # "Undeclared" here means *of a no-op*: with the versions in disagreement the bump is the
    # declaration, and the branch below prints that message over this same list.
    undeclared = _filed(drifts, "arbiter")
    no_op = _filed(drifts, _declared_tag("arbiter"))
    searcher = _filed(drifts, "searcher")

    problems: list[str] = []
    if not agreed:
        problems.append(
            f"stale record: it declares ENGINE_VERSION {recorded_version}, this tree declares "
            f"{computed_version} ({_VERSION_SOURCE})"
        )
    if undeclared:
        names = ", ".join(drift.name for drift in undeclared)
        problems.append(_undeclared_problem(names) if agreed else _older_version_problem(names))
    if no_op:
        names = ", ".join(drift.name for drift in no_op)
        problems.append(
            f"declared no-op arbiter drift: {names}. The changelog entry is there; the record "
            "was not regenerated with it"
        )
    if searcher:
        names = ", ".join(drift.name for drift in searcher)
        problems.append(f"searcher drift (advisory, never blocking): {names}")

    status: Status = "ok"
    if undeclared or no_op or not agreed:
        status = "fail"
    elif searcher:
        status = "warn"
    return Judgement(
        status=status,
        problems=tuple(problems),
        # The strict tier first — and within it what blocks before what is merely stale — so a
        # report reads down from the thing that has to be dealt with.
        drifts=(*undeclared, *no_op, *searcher),
        # Arbiter drift that *is* declared — by a bump already in the tree, or by a no-op entry —
        # is a record that was simply not regenerated yet, and regenerating it is the fix. So
        # what ``--write`` refuses is exactly an undeclared move while the versions agree.
        refuse_write=bool(undeclared) and agreed,
        footer=_footer(computed) if undeclared or no_op else (),
    )


def _tag(name: str, declared: frozenset[str]) -> str:
    """The one word a report prints beside this drift: its tier, and whether it is a declared
    no-op. Searcher drift never blocks, so a declaration over it lifts nothing and is inert."""
    tier = tier_of(name)
    return _declared_tag(tier) if tier == "arbiter" and name in declared else tier


def _declared_tag(tier: Tier) -> str:
    """The tier word with the declaration beside it, spelled in one place for both readings."""
    return f"{tier}, {_NO_OP_TAG}"


def _filed(drifts: tuple[Drift, ...], tag: str) -> tuple[Drift, ...]:
    """The drifts a report files under one word, in the order they were found."""
    return tuple(drift for drift in drifts if drift.tag == tag)


def _undeclared_problem(names: str) -> str:
    """The three outs, with the entry's heading spelled out so the fix is one copy-paste away."""
    return (
        f"arbiter drift with no ENGINE_VERSION bump: {names}. A change here invalidates every "
        f"stored champion comparison — bump ENGINE_VERSION in {_VERSION_SOURCE} in this PR, or "
        f"declare a no-op with a dated entry at the top of {CHANGELOG_PATH} whose heading names "
        f'the component(s), e.g. "## 2026-08-25 — {_COMPONENTS_CLAUSE}: {names} — '
        f'{_BEHAVIOUR_CLAUSE}: {_NO_OP_VALUE}" — or restore the behaviour'
    )


def _older_version_problem(names: str) -> str:
    """The bump is in the tree; only the record has not caught up with it."""
    return (
        f"arbiter drift recorded under an older ENGINE_VERSION: {names}. The bump is there; the "
        "record was not regenerated with it"
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
        "component fails until it is declared — an ENGINE_VERSION bump for a behaviour change, "
        f"or a 'behaviour: unchanged' entry in {CHANGELOG_PATH} for a mechanical one; "
        "searcher-tier drift warns and passes, naming the component and the files that moved. "
        "--write regenerates every other case in one command, and refuses the undeclared one: "
        "an arbiter move must arrive declared."
    ),
    write_help=(
        f"Regenerate {RECORD_PATH} from the tree. Commit it in the same PR as the bump or the "
        f"{CHANGELOG_PATH} entry. Refuses (writing nothing, exit 1) on undeclared arbiter drift."
    ),
    footer=_footer,
)


def build_record(root: Path | None = None) -> dict[str, Any]:
    """The record for ``root``: the declared version, digests, per-file digests, changelog head."""
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
