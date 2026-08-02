"""The site registry — a plain id→:class:`~noctis.eval.site.AgentSite` lookup, declared as code.

There is no registration API here, and that is the design. The declarations are module-level
constants; :data:`DECLARED_SITES` lists them in one literal tuple; :data:`SITES` is the read-only
index over it. Adding a site is editing that tuple — a reviewable diff — never a decorator that
runs at import time, because a registry you can write to at runtime is a registry whose contents
depend on what happened to be imported, and a benchmark whose site set depends on import order is
a benchmark nobody can reproduce.

The lookups take the registry they read as an argument (defaulting to the shipped one), so a
harness can index a scratch set of declarations without a global to reset — the same posture as
the strategy-family registry, which every session builds its own of rather than sharing one.

Nothing in the engine reads this module, and nothing may: the import-isolation guard
(:mod:`noctis.eval.guard`) fails the build if an engine module ever imports the eval layer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

from noctis.eval.site import AgentSite

__all__ = [
    "DECLARED_SITES",
    "SITES",
    "UnknownSite",
    "index_sites",
    "site",
    "sites",
]

# One entry per declared site, in declaration order. Empty until the site declarations land —
# the coder, then the episodic sites (formulate, decide, discover) and distill — each of which is
# a module-level constant appended here as plain code.
DECLARED_SITES: tuple[AgentSite[Any, Any], ...] = ()


class UnknownSite(KeyError):
    """A lookup for a site id nothing declares. Its message names the ids that are declared."""

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


def index_sites(declared: Iterable[AgentSite[Any, Any]]) -> Mapping[str, AgentSite[Any, Any]]:
    """Index declarations by their own id, refusing a duplicate id.

    Two sites answering to one id would make a benchmark record ambiguous about which was run, so
    the collision is refused where it is cheapest to see: at declaration time, naming the id.
    """
    index: dict[str, AgentSite[Any, Any]] = {}
    for declaration in declared:
        if declaration.id in index:
            raise ValueError(f"two sites declare the id {declaration.id!r} — site ids are unique")
        index[declaration.id] = declaration
    return MappingProxyType(index)


SITES: Mapping[str, AgentSite[Any, Any]] = index_sites(DECLARED_SITES)


def sites(
    registry: Mapping[str, AgentSite[Any, Any]] | None = None,
) -> tuple[AgentSite[Any, Any], ...]:
    """Every declared site, in declaration order."""
    return tuple((SITES if registry is None else registry).values())


def site(
    site_id: str, registry: Mapping[str, AgentSite[Any, Any]] | None = None
) -> AgentSite[Any, Any]:
    """The site declared under ``site_id``, or an :class:`UnknownSite` naming what is declared."""
    index = SITES if registry is None else registry
    if site_id not in index:
        declared = ", ".join(sorted(index)) or "no sites"
        raise UnknownSite(f"no site is declared as {site_id!r} — declared sites: {declared}")
    return index[site_id]
