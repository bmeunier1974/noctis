"""Engine identity — never compare two runs across two engines.

Two runs' numbers are comparable only if the same engine produced them. A promotion threshold
moved, a prompt reworded, a shipped mandate profile edited, a seed strategy changed: any of
these shifts results without a single config key differing, and comparing across that boundary
silently attributes an engine change to a config change. So the engine carries an identity, and
it is two things at once:

* :data:`ENGINE_VERSION` — the **declared** comparison key. A plain incrementing integer,
  deliberately decoupled from ``noctis.__version__``: a package release that changes no
  behaviour must not fragment comparison buckets, and a one-line gate change that ships in no
  release must. Bump it in the PR that changes behaviour.
* :func:`fingerprint` — the **computed** drift detector: a digest *per component*, not one
  opaque hash. One hash tells you two runs differ but not whether the difference matters, and
  it invalidates everything on any change. Components let comparison be scoped — two runs with
  different ``prompts`` digests but identical ``gates`` and ``backtest`` digests still have
  comparable scorecards.

:data:`ARBITER_COMPONENTS` draws the one line the rest of the system reads: ``gates`` and
``backtest`` decide *what passes* and *what a number means*, so they bind comparability;
everything else describes how candidates are *found*. It is declared here exactly once — two
copies of that set would eventually disagree, and the disagreement would be silent.

**How the digest is computed, and why.** Raw normalized file content (LF endings, sorted paths),
deliberately *not* AST-stripped of comments: stripping is fragile, and prompt text is
indistinguishable from a comment to any safe automated rule. That over-partitions slightly on a
docstring edit — accepted, because the declared version is the actual comparison key and
under-partitioning is the failure that silently corrupts results. The component map is an
**explicit allowlist of committed files**, never a directory walk, so an operator's gitignored
mandate (``mandate/MANDATE.md``, a custom profile, personal references) can never leak in and
make a fingerprint machine-local — and ``__pycache__`` is excluded by construction. A missing
input yields a ``null`` component with a note, never a crash.

This module is **pure**: source files in, digests out. No config, no clock, no network, no
imports from the rest of ``noctis`` — the election metric a comparable key needs is supplied by
the caller.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

# The behavioural contract version. Plain, incrementing, not semantic, and NOT
# ``noctis.__version__`` (see the module docstring). Bump it in the PR that changes behaviour.
ENGINE_VERSION = 1

# The one line between the judge and the searcher: a change to these two invalidates every
# cross-run champion comparison, so they — not the declared version — bind the comparable key.
# Declared exactly once, here, for every consumer (comparability, resume policy, CI ratchet).
ARBITER_COMPONENTS = frozenset({"gates", "backtest"})

# What counts as behavioural, in one reviewable place. Repo-root-relative POSIX paths, so a
# digest is identical on every platform. Add a file here only if changing it can change a
# result; the review of *this map* is the whole contract.
COMPONENT_PATHS: Mapping[str, tuple[str, ...]] = {
    # Decides what passes. The strictest component.
    "gates": (
        "src/noctis/champions/promotion.py",
        "src/noctis/backtest/scorecard.py",
        "src/noctis/backtest/splits.py",
    ),
    # Decides what a scorecard number *means* — including the fill/cost model the
    # execution-realistic validation stage and the coarse pre-filter both price trades with.
    "backtest": (
        "src/noctis/backtest/pipeline.py",
        "src/noctis/backtest/validate.py",
        "src/noctis/backtest/candidate.py",
        "src/noctis/backtest/prefilter.py",
        "src/noctis/broker/seam.py",
        "src/noctis/broker/paper.py",
    ),
    # Decides how candidates are found (the tool registry lives in ``tools.py``).
    "research": (
        "src/noctis/research/agent.py",
        "src/noctis/research/driver.py",
        "src/noctis/research/tools.py",
        "src/noctis/research/episode.py",
        "src/noctis/research/sweep.py",
    ),
    # Decides what the model is told.
    "prompts": (
        "src/noctis/research/prompt.py",
        "src/noctis/research/briefings.py",
        "src/noctis/research/contract_sheet.py",
        "src/noctis/research/digests.py",
        "src/noctis/research/ideation.py",
    ),
    # The COMMITTED mandate scaffold only — the shipped steering personalities. The operator's
    # own MANDATE.md, custom personalities and references are gitignored and stay out by name
    # (they are already pinned per-run, frozen verbatim in the run record's inputs). The two
    # READMEs (mandate/, strategies/) are out too: no session ever reads them — they document
    # the scaffold for a human, so editing one changes nothing a run does.
    "profiles": (
        "mandate/MANDATE.md.example",
        "mandate/tune-first.md",
        "mandate/profiles/aggressive.md",
        "mandate/profiles/conservative.md",
        "mandate/profiles/long-term.md",
        "mandate/profiles/sector-specialist.md",
        "mandate/profiles/short-term.md",
    ),
    # The read-only library input every run starts from.
    "seeds": (
        "strategies/TEMPLATE.py",
        "strategies/donchian_breakout.py",
        "strategies/rsi_meanrev.py",
        "strategies/sma_crossover.py",
    ),
    # The starting condition of every run's memory.
    "memory_seed": ("MEMORY.seed.md",),
    # Changes what is recorded. The run-record schema module does not exist yet (#143); until
    # it lands this component is null-with-a-note, which is exactly the missing-input rule.
    "schema": ("src/noctis/reporting/schema.py",),
}

# A truncated SHA-256 (64 bits of it): long enough that two distinct engines colliding is not a
# realistic concern, short enough to read in a terminal and paste into an issue.
_DIGEST_CHARS = 16


@dataclass(frozen=True)
class ComponentFingerprint:
    """One component's identity: its digest, the inputs actually hashed, and why it is null.

    ``digest`` is ``None`` — the missing-input rule — whenever any mapped path is absent, and
    then ``note`` names what was missing. A partial digest is worse than none: it would look
    like an identity while quietly meaning "some of the engine".
    """

    name: str
    digest: str | None
    files: tuple[str, ...]
    note: str | None = None


@dataclass(frozen=True)
class EngineFingerprint:
    """The declared version plus one digest per behavioural component."""

    engine_version: int
    components: Mapping[str, ComponentFingerprint]

    def digest(self, component: str) -> str | None:
        """This component's digest, or ``None`` when an input was missing."""
        return self.components[component].digest

    def note(self, component: str) -> str | None:
        """Why this component is null, or ``None`` when it is fine."""
        return self.components[component].note

    def digests(self) -> dict[str, str | None]:
        """The plain component → digest map (the shape a record or a diff wants)."""
        return {name: comp.digest for name, comp in self.components.items()}


class ComparableKey(NamedTuple):
    """The tuple two runs must match on before their numbers may be pooled, ranked or plotted.

    The ``election_metric`` is in it for the reason promotion already treats a differently
    scored champion as *stale*: numbers under different metrics are in different units and were
    never comparable. The two digests — not the declared version — carry the strict guarantee,
    because a digest cannot be forgotten in review.
    """

    engine_version: int
    gates_digest: str | None
    backtest_digest: str | None
    election_metric: str

    def __str__(self) -> str:
        """``1|f63d47b7b9604ab1|3ba3e0bf1c97134f|sharpe`` — one greppable bucket label."""
        return "|".join("null" if part is None else str(part) for part in self)


def fingerprint(root: Path | None = None) -> EngineFingerprint:
    """Compute the per-component fingerprint of the engine rooted at ``root``.

    ``root`` is the repo root every path in :data:`COMPONENT_PATHS` is resolved against;
    it defaults to this checkout (see :func:`default_root`) and is injectable so a caller —
    or a test — can fingerprint another tree.
    """
    base = default_root() if root is None else Path(root)
    return EngineFingerprint(
        engine_version=ENGINE_VERSION,
        components={
            name: _component_fingerprint(name, base, paths)
            for name, paths in COMPONENT_PATHS.items()
        },
    )


def comparable_key(
    election_metric: str, engine_fingerprint: EngineFingerprint | None = None
) -> ComparableKey:
    """``(engine_version, gates_digest, backtest_digest, election_metric)``.

    Two runs may have their champion and scorecard numbers compared **only** if this tuple
    matches. ``election_metric`` is supplied by the caller (from ``promotion.metric``) so this
    module never reads configuration.
    """
    fp = fingerprint() if engine_fingerprint is None else engine_fingerprint
    return ComparableKey(
        engine_version=fp.engine_version,
        gates_digest=fp.digest("gates"),
        backtest_digest=fp.digest("backtest"),
        election_metric=election_metric,
    )


def compare(a: EngineFingerprint, b: EngineFingerprint) -> set[str]:
    """The names of the components whose digests differ between two fingerprints.

    A component present in only one side counts as differing; two nulls do not (neither side
    could identify it, so nothing is known to have moved).
    """
    names = set(a.components) | set(b.components)
    return {name for name in names if _digest_of(a, name) != _digest_of(b, name)}


def default_root() -> Path:
    """The repo root of this checkout: the nearest ancestor holding ``pyproject.toml``.

    Installed from a wheel there is no checkout to find, and the fallback simply will not
    resolve the mapped paths — which degrades to null components with notes rather than a
    crash, the same rule any other missing input takes.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[2]


def _component_fingerprint(name: str, root: Path, rel_paths: Iterable[str]) -> ComponentFingerprint:
    """Hash one component's allowlisted files: sorted paths, LF-normalized content, all or none."""
    hasher = hashlib.sha256()
    present: list[str] = []
    missing: list[str] = []
    for rel in sorted(rel_paths):
        path = root.joinpath(*rel.split("/"))
        if not path.is_file():
            missing.append(rel)
            continue
        present.append(rel)
        # The path rides into the hash beside its content, so a rename moves the digest too.
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_normalized(path.read_bytes()))
        hasher.update(b"\0")
    if missing:
        return ComponentFingerprint(
            name=name,
            digest=None,
            files=tuple(present),
            note=f"missing input(s): {', '.join(missing)}",
        )
    return ComponentFingerprint(
        name=name, digest=hasher.hexdigest()[:_DIGEST_CHARS], files=tuple(present)
    )


def _normalized(raw: bytes) -> bytes:
    """LF line endings, so a CRLF checkout fingerprints identically to an LF one."""
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _digest_of(fp: EngineFingerprint, name: str) -> tuple[bool, str | None]:
    component = fp.components.get(name)
    return (component is not None, component.digest if component is not None else None)
