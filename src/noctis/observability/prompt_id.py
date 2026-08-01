"""Prompt identity — a result must be able to name the prompt that produced it.

Prompt text lives as module constants: a wording change ships in a diff nobody flags, and three
weeks later a scorecard cannot say what the model was actually asked. This module is the missing
identity — a content hash **per LLM call site**, computed from that site's committed prompt-bearing
source files.

**Per site, not per engine, and that is the whole design.** One hash over every prompt in the repo
would tell you two sessions differ without telling you whether the difference could matter: a
reworded distillation prompt would invalidate the coder's identity, and nothing would be
comparable to anything for long. So each site — ``author``, ``conversation``, ``episodic``,
``briefings``, ``ideation``, ``distill`` — carries its own digest, and a future benchmark record
names the one site it is grading (:func:`site_digest` is that read).

This is deliberately a **separate artifact** from :mod:`~noctis.observability.engine_id`. Prompts
and arbiter behaviour drift on different clocks and are consumed by different keys: a prompt
rewrite must not read as "the judge moved", and a promotion-threshold change must not read as "the
model was told something new". Neither identity is defined in terms of the other; the *hashing
rule* is shared (:func:`~noctis.observability.engine_id.file_digest`) precisely so two answers to
"what is the content of this committed file" can never disagree.

**A site digest is the digest of its files' digests.** Sorted repo-relative POSIX paths, each
paired with the same LF-normalized content hash the engine fingerprint takes, then hashed together
with the path riding beside the digest so a rename moves the site too. That composition is worth
stating because it makes the committed record **self-verifying**: a reader holding
``prompt_fingerprint.json`` can recompute every site digest from the per-file rows it already
carries, without the sources.

**What a site's assets are**: the modules that author or render text for it — including the shared
fact-renderer ``research/digests.py``, which is listed under *every* site that assembles it. That
over-partitions (a market-digest edit moves four sites at once), and over-partitioning is the
accepted direction: a site whose assembled text changed must never keep its old identity, while
under-partitioning is exactly the silence this module exists to end. The map is an **explicit
allowlist of committed files**, never a directory walk, so nothing gitignored can make a digest
machine-local. A missing input yields a ``null`` site with a note, never a crash.

This module is **pure**: source files in, digests out. No config, no clock, no network. The CI
check that holds the repo to a committed statement of these digests is
:mod:`noctis.observability.prompt_ratchet`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from noctis.observability.engine_id import default_root, file_digest

__all__ = [
    "SITE_ASSETS",
    "PromptFingerprint",
    "SiteFingerprint",
    "compare",
    "default_root",
    "file_digest",
    "fingerprint",
    "site_digest",
]

# The shared fact-renderer: every site whose prompt embeds a market digest, library index or
# champion board renders it through here, so it is listed under each of them by name.
_SHARED_DIGESTS = "src/noctis/research/digests.py"

# What counts as one call site's prompt, in one reviewable place. Repo-root-relative POSIX paths,
# so a digest is identical on every platform. Add a file here only if changing it can change what
# a model is told; the review of *this map* is the whole contract.
SITE_ASSETS: Mapping[str, tuple[str, ...]] = {
    # The coder site: the authoring brief and the contract sheet it must satisfy.
    "author": (
        "src/noctis/research/author.py",
        "src/noctis/research/contract_sheet.py",
        _SHARED_DIGESTS,
    ),
    # The rendered briefings that are the episodic stages' user turns.
    "briefings": (
        "src/noctis/research/briefings.py",
        _SHARED_DIGESTS,
    ),
    # The conversation loop's system prompt.
    "conversation": (
        "src/noctis/research/prompt.py",
        _SHARED_DIGESTS,
    ),
    # The memory distiller's one summarization prompt.
    "distill": ("src/noctis/research/distill.py",),
    # The episodic driver's per-stage system texts and emit contracts.
    "episodic": (
        "src/noctis/research/driver.py",
        _SHARED_DIGESTS,
    ),
    # The seeded-idea prompt, web search included.
    "ideation": ("src/noctis/research/ideation.py",),
}

# A truncated SHA-256, the same width the engine fingerprint prints: long enough that a collision
# is not a realistic concern, short enough to read in a terminal and paste into an issue.
_DIGEST_CHARS = 16


@dataclass(frozen=True)
class SiteFingerprint:
    """One call site's prompt identity: its digest, the files hashed, and why it is null.

    ``digest`` is ``None`` — the missing-input rule — whenever any mapped path is absent, and then
    ``note`` names what was missing. A partial digest is worse than none: it would look like an
    identity while quietly meaning "some of the prompt".
    """

    name: str
    digest: str | None
    files: tuple[str, ...]
    note: str | None = None


@dataclass(frozen=True)
class PromptFingerprint:
    """One digest per LLM call site — the whole prompt surface of a checkout."""

    sites: Mapping[str, SiteFingerprint]

    def digest(self, site: str) -> str | None:
        """This site's digest, or ``None`` when an input was missing."""
        return self.sites[site].digest

    def note(self, site: str) -> str | None:
        """Why this site is null, or ``None`` when it is fine."""
        return self.sites[site].note

    def digests(self) -> dict[str, str | None]:
        """The plain site → digest map (the shape a record or a diff wants)."""
        return {name: site.digest for name, site in self.sites.items()}


def fingerprint(root: Path | None = None) -> PromptFingerprint:
    """Compute the per-site prompt fingerprint of the tree at ``root``.

    ``root`` is the repo root every path in :data:`SITE_ASSETS` is resolved against; it defaults
    to this checkout (:func:`~noctis.observability.engine_id.default_root`) and is injectable so a
    caller — or a test — can fingerprint another tree.
    """
    base = default_root() if root is None else Path(root)
    return PromptFingerprint(
        sites={name: _site_fingerprint(name, base, paths) for name, paths in SITE_ASSETS.items()}
    )


def site_digest(site: str, root: Path | None = None) -> str | None:
    """One call site's prompt hash — the pure read a benchmark record's key is built from.

    ``None`` when one of the site's assets is missing, and a :class:`KeyError` for a site nobody
    has heard of: a typo'd name must never read as "this site has no prompt".
    """
    if site not in SITE_ASSETS:
        raise KeyError(f"unknown prompt site: {site!r} (known: {', '.join(sorted(SITE_ASSETS))})")
    base = default_root() if root is None else Path(root)
    return _site_fingerprint(site, base, SITE_ASSETS[site]).digest


def compare(a: PromptFingerprint, b: PromptFingerprint) -> set[str]:
    """The names of the sites whose digests differ between two fingerprints.

    A site present in only one side counts as differing; two nulls do not (neither side could
    identify it, so nothing is known to have moved) — the same rule the engine fingerprint takes.
    """
    names = set(a.sites) | set(b.sites)
    return {name for name in names if _digest_of(a, name) != _digest_of(b, name)}


def _site_fingerprint(name: str, root: Path, rel_paths: Iterable[str]) -> SiteFingerprint:
    """Hash one site: sorted paths beside their file digests, all or none."""
    hasher = hashlib.sha256()
    present: list[str] = []
    missing: list[str] = []
    for rel in sorted(rel_paths):
        digest = file_digest(root.joinpath(*rel.split("/")))
        if digest is None:
            missing.append(rel)
            continue
        present.append(rel)
        # The path rides into the hash beside its content digest, so a rename moves the site too.
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(digest.encode("utf-8"))
        hasher.update(b"\0")
    if missing:
        return SiteFingerprint(
            name=name,
            digest=None,
            files=tuple(present),
            note=f"missing input(s): {', '.join(missing)}",
        )
    return SiteFingerprint(
        name=name, digest=hasher.hexdigest()[:_DIGEST_CHARS], files=tuple(present)
    )


def _digest_of(fp: PromptFingerprint, name: str) -> tuple[bool, str | None]:
    site = fp.sites.get(name)
    return (site is not None, site.digest if site is not None else None)
