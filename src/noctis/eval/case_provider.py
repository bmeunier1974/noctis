"""The YAML filesystem provider — the one place in the eval layer that reads a corpus off disk.

:class:`YamlCaseProvider` is the v1 :class:`~noctis.eval.case.CaseProvider`: one directory per
site under a cases root, one case per ``*.yaml`` file, the file's stem as the case id. That layout
is the point rather than an implementation detail — a case is then a small file a reviewer reads in
a diff, a curator adds by writing one, and git tracks by name — while the protocol keeps the runner,
the metrics and the record from ever learning it.

**Everything it reads, it validates.** Every file goes through :func:`~noctis.eval.case.parse_case`
with the site registry's declared ids, so a case naming a site nothing declares, carrying an
expected output, or missing a field is refused here rather than counted later. Three refusals are
the provider's own, because they are facts about the *layout* and only it can see them:

* a **site id nothing declares** — the registry's own :class:`~noctis.eval.registry.UnknownSite`,
  looked up through the registry the provider was given exactly as every other eval-layer lookup
  is, so a scratch set of declarations needs no global to reset;
* a **case filed under the wrong site** — a ``decide`` case sitting in ``coder/`` would otherwise
  be benchmarked as a coder case, which is invariant "every result is attributable to exactly one
  site" broken by a typo;
* a **missing case directory** — refused, not read as an empty corpus. A benchmark over zero cases
  that reports itself as a benchmark is the one failure a corpus loader must not have. An
  *existing but empty* directory loads nothing and says so by returning nothing: that is a corpus
  somebody has started, visible in git, not a root path spelled wrong.

A file saved as ``*.yml`` is refused for the same reason rather than skipped: silently unread is
how a case stops being measured without anyone noticing. Files are read in sorted order, so the
same directory always yields the same tuple — the property #197's deterministic split stands on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from noctis.eval.case import Case, MalformedCase, parse_case
from noctis.eval.registry import SITES, site
from noctis.eval.site import AgentSite

__all__ = [
    "CASE_SUFFIX",
    "CasePaths",
    "CaseRootSpec",
    "MissingCorpus",
    "YamlCaseProvider",
    "missing_corpus",
]

# One case per file, and the extension is exact: a near-miss is refused, never quietly unread.
CASE_SUFFIX = ".yaml"


class MissingCorpus(FileNotFoundError):
    """No case directory where a site's corpus was expected. The message names every path tried."""


@dataclass(frozen=True)
class CasePaths:
    """The corpus tiers a site is read from, **lowest precedence first**.

    A corpus has the same two halves everything else committed in this repo has: the curated cases
    a review shipped (the repo's ``cases/``, read-only input) and the operator's own (mined,
    harvested, machine-written — under ``<workspace>/cases/``). Reading both is what makes the
    shipped buckets reachable without a copy, and copying is what this exists to avoid: a copied
    corpus goes stale silently, and two people then run "the smoke tier" over two populations while
    the digest is the only thing that would have told them.

    Precedence is the strategy library's, for the same reason: a later tier overrides an earlier
    one by case id, so an operator can shadow a shipped case locally and a shipped case can never
    silently replace one of theirs. Writers (the DECIDE miner, ``stamp_splits``) take a single root
    and are unaffected — the tiers are a *reading* contract.
    """

    roots: tuple[Path, ...]

    @classmethod
    def from_single_root(cls, root: str | Path) -> CasePaths:
        """One root, which is what a bare path has always meant — how legacy callers coerce."""
        return cls((Path(root),))

    @classmethod
    def coerce(cls, value: CaseRootSpec) -> CasePaths:
        """A bare path means the historical single-root layout."""
        return value if isinstance(value, CasePaths) else cls.from_single_root(value)

    def directories(self, site_id: str) -> tuple[Path, ...]:
        """This site's directory in every tier that has one, lowest precedence first.

        A tier without the directory is not a defect — an operator whose workspace has never been
        mined has no local tier, and a checkout is not obliged to ship a corpus for every site.
        Only *all* of them missing is a refusal, and :func:`missing_corpus` names them all.
        """
        return tuple(Path(root) / site_id for root in self.roots if (Path(root) / site_id).is_dir())


#: What every corpus reader accepts where it used to take a bare directory.
CaseRootSpec = CasePaths | str | Path


def missing_corpus(site_id: str, paths: CasePaths) -> MissingCorpus:
    """The refusal for a site no tier holds, naming each tier — a root spelled wrong is visible."""
    tried = ", ".join(str(Path(root) / site_id) for root in paths.roots)
    return MissingCorpus(
        f"no case directory for site {site_id!r} at {tried} — a benchmark over an "
        "absent corpus is a number nobody can trust"
    )


@dataclass(frozen=True)
class YamlCaseProvider:
    """Cases loaded from ``<cases_root>/<site_id>/*.yaml``, validated file by file.

    ``cases_root`` is a :class:`CasePaths` (the committed tier, then the workspace's) or a bare
    path, which still means the single root it always did. ``registry`` is the site index the
    provider cross-checks ids against, defaulting to the shipped declarations — passed in, never
    reached for globally, so a test or a harness can load a corpus for a scratch set of sites.
    """

    cases_root: CaseRootSpec
    registry: Mapping[str, AgentSite[Any, Any]] | None = None

    def load(self, site_id: str) -> tuple[Case, ...]:
        """Every case declared for ``site_id``, in case-id order, or a loud refusal.

        Tiers are read lowest precedence first and folded by case id, so a workspace case shadows a
        committed one of the same name and the result is ordered by id — which, for the single-root
        layout, is the file-name order this has always returned.
        """
        index = SITES if self.registry is None else self.registry
        site(site_id, index)
        paths = CasePaths.coerce(self.cases_root)
        directories = paths.directories(site_id)
        if not directories:
            raise missing_corpus(site_id, paths)
        declared = frozenset(index)
        found: dict[str, Case] = {}
        for directory in directories:
            stray = next(iter(sorted(directory.glob("*.yml"))), None)
            if stray is not None:
                raise MalformedCase(
                    f"{stray}: a case file is named '<case id>{CASE_SUFFIX}' — rename it rather "
                    "than leave it unread"
                )
            for path in sorted(directory.glob(f"*{CASE_SUFFIX}")):
                case = self._case(path, site_id, declared)
                found[case.case_id] = case
        return tuple(found[case_id] for case_id in sorted(found))

    def _case(self, path: Path, site_id: str, declared: frozenset[str]) -> Case:
        """One file as a validated case, refused by path on bad YAML or a misfiled site."""
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise MalformedCase(f"{path}: not valid YAML — {error}") from error
        case = parse_case(document, case_id=path.stem, source=str(path), declared_site_ids=declared)
        if case.site_id != site_id:
            raise MalformedCase(
                f"{path}: site_id {case.site_id!r} does not match its directory {site_id!r} — a "
                "case lives in the directory of the site it exercises"
            )
        return case
