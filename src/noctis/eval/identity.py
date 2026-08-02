"""Per-site identity — the declared version paired with that site's prompt-asset hash.

A benchmark result is only comparable to another when two things about the site that produced it
were the same: the *shape* of the judgment asked (its contract generation) and the *text* it was
asked with. Both facts already exist — :attr:`~noctis.eval.site.AgentSite.version` is hand-bumped
on the declaration, and :mod:`noctis.observability.prompt_id` computes a content hash per prompt
asset group — but they live on opposite sides of the eval boundary, and a record builder that
reached for each of them itself would be a second place that knows how sites map onto prompt
assets. This module is that knowledge, once: :func:`site_identity` is **the only bridge** between
the site registry and the prompt-fingerprint module, and it lives on the eval side because the
engine has no business reading a benchmark's identity of it.

**The map is the crux, so it is written out.** The eval layer's site ids (``coder``, ``formulate``,
``decide``, ``discover``, ``distill``) and the prompt-asset groups (``author``, ``briefings``,
``conversation``, ``distill``, ``episodic``, ``ideation``) are two honest partitions of different
things — one names an LLM *judgment*, the other names a set of committed *files* — and they are
deliberately not one-to-one. :data:`PROMPT_ASSET_GROUPS` states the correspondence explicitly, and
a site maps to **every** group whose text its prompt is assembled from:

* the coder's brief and contract sheet are the ``author`` group, exactly;
* the memory distiller's one template is the ``distill`` group, exactly;
* each episodic site is assembled from *two* groups — ``briefings`` (the rendered user turn its
  ``render`` forwards to) and ``episodic`` (the driver's per-stage system texts). Mapping such a
  site to only one of them would let half its prompt be reworded while its identity stood still,
  which is precisely the silence the prompt-asset hash exists to end.

**The three episodic sites therefore share one hash, and that is honest.** They are assembled from
the same prompt-bearing modules, so no hash over those modules can tell them apart; what tells them
apart is their own ``version`` and their contract. A record folds the pair, never the hash alone.

**Which read of the hash, and why the tree.** :func:`~noctis.observability.prompt_id.site_digest`
is the read that module names for exactly this purpose ("the pure read a benchmark record's key is
built from"), and it recomputes from the source tree. The committed ``prompt_fingerprint.json`` is
the *ratchet's* artifact — the baseline CI compares a checkout against — and reading it here would
make a result's identity depend on whether someone had regenerated the record, which is a fact
about bookkeeping, not about what the model was told. Recomputing states what the running checkout
would actually send; the prompt ratchet is what keeps that equal to the declared record, and a tree
where the two disagree is a tree CI already fails.

**The hashing rule is prompt_id's own, by identity, never a copy.** A site whose file set spans two
groups is hashed by the same function that hashes a group — sorted repo-relative paths beside the
same LF-normalized file digests, all-or-none on a missing input — so a single-group site's identity
hash *is* the group digest the committed record carries, greppable and unchanged. Re-deriving that
composition here would be a second answer to "what is the content of these committed files", and
two such answers eventually disagree.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from noctis.eval.registry import site, sites
from noctis.eval.site import AgentSite
from noctis.observability import prompt_id

__all__ = ["PROMPT_ASSET_GROUPS", "SiteIdentity", "site_identities", "site_identity"]

# Eval site id → the prompt-asset groups its prompt is assembled from, in one reviewable place.
# Every group name is a key of :data:`noctis.observability.prompt_id.SITE_ASSETS`; a declared site
# missing a row here is refused rather than hashed to null (see :func:`site_identity`), because a
# new site arriving without its prompt assets named is a gap, not an identity.
PROMPT_ASSET_GROUPS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        # The authoring brief plus the contract sheet it must satisfy — the coder's whole ask.
        "coder": ("author",),
        # The three episodic stages: the rendered briefing (user turn) and the driver's system
        # texts. Shared, because they really are assembled from the same modules.
        "formulate": ("briefings", "episodic"),
        "decide": ("briefings", "episodic"),
        "discover": ("briefings", "episodic"),
        # The memory distiller's single summarization template.
        "distill": ("distill",),
    }
)


@dataclass(frozen=True)
class SiteIdentity:
    """What makes one site's results comparable: its declared generation and its prompt's hash.

    ``prompt_asset_hash`` is ``None`` — the prompt module's all-or-none rule — when any of the
    site's assets is missing from the tree; a partial hash would look like an identity while
    meaning "some of the prompt". ``prompt_asset_groups`` names where the hash came from, so a
    reader holding a record can recompute it from the committed per-file digests.
    """

    site_id: str
    version: str
    prompt_asset_groups: tuple[str, ...]
    prompt_asset_hash: str | None


def site_identity(
    site_id: str,
    root: Path | None = None,
    registry: Mapping[str, AgentSite[Any, Any]] | None = None,
) -> SiteIdentity:
    """One site's identity: the version it declares and the hash of its prompt assets.

    ``root`` is the repo root the prompt assets are read from (this checkout by default, injectable
    so a harness — or a test — can identify another tree), and ``registry`` the site index to look
    the declaration up in, defaulting to the shipped one exactly as
    :func:`~noctis.eval.registry.site` does.

    Raises :class:`~noctis.eval.registry.UnknownSite` for an id nothing declares, and
    :class:`ValueError` for a site that is declared but names no prompt-asset groups.
    """
    declaration = site(site_id, registry)
    groups = PROMPT_ASSET_GROUPS.get(declaration.id)
    if not groups:
        raise ValueError(
            f"site {declaration.id!r} maps to no prompt-asset group — add its row to "
            "PROMPT_ASSET_GROUPS naming the groups its prompt is assembled from"
        )
    return SiteIdentity(
        site_id=declaration.id,
        version=declaration.version,
        prompt_asset_groups=groups,
        prompt_asset_hash=_assets_digest(declaration.id, groups, root),
    )


def site_identities(
    root: Path | None = None,
    registry: Mapping[str, AgentSite[Any, Any]] | None = None,
) -> tuple[SiteIdentity, ...]:
    """Every declared site's identity, in declaration order."""
    return tuple(site_identity(declaration.id, root, registry) for declaration in sites(registry))


def _assets_digest(site_id: str, groups: tuple[str, ...], root: Path | None) -> str | None:
    """Hash the union of the mapped groups' assets with the prompt module's own rule.

    The rule is not restated here: :func:`noctis.observability.prompt_id._site_fingerprint` is
    called with this site's file set, the way the coder and distill declarations bind production's
    private prompt objects rather than copying them. A single-group site therefore hashes to the
    exact digest the committed record carries for that group.
    """
    base = prompt_id.default_root() if root is None else Path(root)
    files = {rel for group in groups for rel in prompt_id.SITE_ASSETS[group]}
    return prompt_id._site_fingerprint(site_id, base, files).digest
