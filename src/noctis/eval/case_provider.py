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

__all__ = ["CASE_SUFFIX", "MissingCorpus", "YamlCaseProvider"]

# One case per file, and the extension is exact: a near-miss is refused, never quietly unread.
CASE_SUFFIX = ".yaml"


class MissingCorpus(FileNotFoundError):
    """No case directory where a site's corpus was expected. The message names the path."""


@dataclass(frozen=True)
class YamlCaseProvider:
    """Cases loaded from ``<cases_root>/<site_id>/*.yaml``, validated file by file.

    ``registry`` is the site index the provider cross-checks ids against, defaulting to the
    shipped declarations — passed in, never reached for globally, so a test or a harness can load
    a corpus for a scratch set of sites.
    """

    cases_root: Path
    registry: Mapping[str, AgentSite[Any, Any]] | None = None

    def load(self, site_id: str) -> tuple[Case, ...]:
        """Every case declared for ``site_id``, in file-name order, or a loud refusal."""
        index = SITES if self.registry is None else self.registry
        site(site_id, index)
        directory = Path(self.cases_root) / site_id
        if not directory.is_dir():
            raise MissingCorpus(
                f"no case directory for site {site_id!r} at {directory} — a benchmark over an "
                "absent corpus is a number nobody can trust"
            )
        stray = next(iter(sorted(directory.glob("*.yml"))), None)
        if stray is not None:
            raise MalformedCase(
                f"{stray}: a case file is named '<case id>{CASE_SUFFIX}' — rename it rather than "
                "leave it unread"
            )
        declared = frozenset(index)
        paths = sorted(directory.glob(f"*{CASE_SUFFIX}"))
        return tuple(self._case(path, site_id, declared) for path in paths)

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
