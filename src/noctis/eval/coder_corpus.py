"""The coder corpus: four bucket directories, one stamped split, and the tool that stamps it.

The coder site is the first one this repo ships a *committed* corpus for, and a committed corpus
raises two questions the generic eval core deliberately does not answer. Where do the files live so
that a curated bucket can ship to every user while a harvested one stays the operator's? And who
decides the tuning/holdout split of a corpus that grows one reviewed pull request at a time? This
module answers both, and nothing else: it is the coder half of :mod:`noctis.eval.case_provider`
(which stays the generic flat-directory reader, untouched) plus the authoring-time tool that freezes
a split into the files themselves.

**Layout: one directory per bucket, under the site's own directory.**

```
cases/coder/
├─ README.md          the case format, the axes, the split rule            (committed)
├─ edge/              deliberately hard briefs, one axis at its extreme    (committed)
├─ canary/            briefs so plain a failure indicts the harness        (committed)
├─ field/             harvested from an operator's own runs                (LOCAL, gitignored)
└─ replay/            an operator's own known gate rejections              (LOCAL, gitignored)
```

The generic provider reads ``<cases_root>/<site_id>/*.yaml`` — flat — and a flat coder corpus was
the alternative: the bucket already rides as a ``bucket:<name>`` tag, so the directories carry no
information the files do not. It was rejected on the one thing a directory can do that a tag cannot:
``.gitignore`` matches *paths*. Committed-versus-local is a per-file decision in a flat layout —
an allowlist line per case id, or a naming convention a curator can typo into a leaked corpus —
while with directories it is four lines that never change again, the deny-by-default shape the
``mandate/`` scaffold already uses. So :class:`CoderCaseProvider` walks the buckets and hands the
generic :class:`~noctis.eval.corpus.Corpus` the cases it finds; the runner, the metrics and the
record still meet a :class:`~noctis.eval.case.CaseProvider` and learn nothing about either layout.

Everything the loader can see about the *layout*, it refuses: a case filed in a directory that
disagrees with its own ``bucket:`` tag (the label a benchmark reports over would be a lie), a case
sitting beside the buckets rather than in one, a ``.yml`` near-miss, a sub-directory that is not a
declared bucket — all of them the same failure, a case that is silently not measured. Everything
about the *content* is the schema's: each file goes through
:func:`~noctis.eval.coder_case.parse_coder_case`, so a committed case that has rotted fails CI
rather than skewing a number.

**The split is stamped into the files, and reading never re-deals it.** The generic corpus deals a
stratified split deterministically, but *deterministic* is not *frozen*: its own contract says the
assignment depends on the whole stratum, so adding a case can move an unassigned one. A corpus that
grows a case a week is exactly the corpus that would keep re-dealing. So the deal happens once, at
authoring time (:func:`stamp_splits`, run by a curator when new cases land), and the answer is
written into each file's ``split:`` field — after which every loader honours what the file says,
because :meth:`~noctis.eval.case.Case.assigned_to` refuses reassignment and the corpus counts frozen
cases toward their stratum's quota. The committed files therefore carry their halves in the diff,
where a reviewer can see one move.

**Bucket is part of the stratum key, by partition.** A difficulty stratum means "cases of the same
kind", and two cases with identical axes in different buckets are not the same kind: a canary and a
field case share nothing but their labels. The generic corpus stratifies on the ``difficulty``
mapping alone, and folding a synthetic ``bucket`` axis into that mapping was the alternative —
rejected because the seven axes are a closed set the schema validates, and a corpus whose files
carry an eighth axis nobody declared is a corpus that reads differently through the two paths. So
:func:`deal_splits` **partitions by bucket first** and runs the generic deal within each: the
composed key is ``(bucket, difficulty axes)``, the rule inside a bucket is the corpus's own, and
each bucket lands on the holdout share by itself — which is also what a reader wants, since a
holdout with no canaries in it is not a holdout of this corpus.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from noctis.eval.case import Case, MalformedCase, Split, parse_case
from noctis.eval.case_provider import CASE_SUFFIX, MissingCorpus
from noctis.eval.coder_case import BUCKET_TAG_PREFIX, Bucket, coder_payload, parse_coder_case
from noctis.eval.corpus import Corpus
from noctis.eval.registry import SITES, site
from noctis.eval.site import AgentSite

__all__ = [
    "CODER_SITE_ID",
    "COMMITTED_BUCKETS",
    "LOCAL_BUCKETS",
    "CoderCaseProvider",
    "bucket_of",
    "deal_splits",
    "load_coder_corpus",
    "stamp_splits",
]

#: The one site this loader reads. A corpus measures exactly one site, and this one is the coder's.
CODER_SITE_ID = "coder"

#: The buckets a reviewed corpus ships to every user: curated by hand, safe to read in a diff.
COMMITTED_BUCKETS: tuple[Bucket, ...] = (Bucket.EDGE, Bucket.CANARY)

#: The buckets harvested from an operator's own runs — their briefs, their symbols, their run ids.
#: Gitignored, and never re-included: a corpus mined from somebody's runs is theirs.
LOCAL_BUCKETS: tuple[Bucket, ...] = (Bucket.FIELD, Bucket.REPLAY)


@dataclass(frozen=True)
class CoderCaseProvider:
    """Coder cases loaded from ``<cases_root>/coder/<bucket>/*.yaml``, validated file by file.

    A :class:`~noctis.eval.case.CaseProvider` like the flat YAML one, and deliberately not a
    subclass of it: the two read different layouts, and the protocol is what they share.
    ``registry`` is the site index ids are cross-checked against, passed in rather than reached for
    globally so a test or a harness can load a corpus for a scratch set of sites.
    """

    cases_root: Path
    registry: Mapping[str, AgentSite[Any, Any]] | None = None

    def load(self, site_id: str) -> tuple[Case, ...]:
        """Every coder case in every bucket, in case-id order, or a loud refusal."""
        if site_id != CODER_SITE_ID:
            raise ValueError(
                f"the coder corpus loader reads site {CODER_SITE_ID!r}, asked for {site_id!r} — "
                "a corpus measures exactly one site, and this layout is the coder's"
            )
        return tuple(case for _, case in _read(self.cases_root, self.registry))


def load_coder_corpus(
    cases_root: Path, registry: Mapping[str, AgentSite[Any, Any]] | None = None
) -> Corpus:
    """The whole coder corpus as one :class:`~noctis.eval.corpus.Corpus`, splits as stamped.

    The cases arrive already assigned (every committed file carries its ``split:``), so the corpus
    deals nothing here — it holds them, refuses a repeated id across buckets, and hashes the result.
    """
    return Corpus(
        site_id=CODER_SITE_ID,
        cases=CoderCaseProvider(cases_root=cases_root, registry=registry).load(CODER_SITE_ID),
    )


def bucket_of(case: Case) -> Bucket:
    """The bucket a case declares, read through the coder schema's own validation."""
    return coder_payload(case).bucket


def deal_splits(cases: Sequence[Case]) -> dict[str, Split]:
    """The tuning/holdout assignment for ``cases``, keyed by case id: the authoring-time deal.

    Pure, and the only place the stratum key is composed: cases are partitioned by bucket and each
    partition is dealt by the generic corpus's own rule (stratify on the difficulty axes, order by
    the hash of the case id, cut at the holdout share). A case that already carries a split keeps
    it and counts toward its stratum's quota, exactly as it does everywhere else.
    """
    partitioned: dict[Bucket, list[Case]] = {}
    for case in cases:
        partitioned.setdefault(bucket_of(case), []).append(case)
    return {
        case.case_id: case.split
        for bucket in Bucket
        for case in Corpus(CODER_SITE_ID, tuple(partitioned.get(bucket, ()))).cases
        if case.split is not None
    }


def stamp_splits(
    cases_root: Path, registry: Mapping[str, AgentSite[Any, Any]] | None = None
) -> tuple[Path, ...]:
    """Freeze the split into every case file that declares none, and say which files moved.

    The authoring-time half of the contract, run by a curator once when new cases land — never by a
    benchmark, never on a schedule. Idempotent by construction: a file that already declares a
    ``split:`` is left byte for byte alone (its assignment is frozen, and frozen is frozen), so a
    second run stamps nothing and returns ``()``. The stamp is one appended line, so a file keeps
    its own field order, comments and formatting.
    """
    files = _read(cases_root, registry)
    dealt = deal_splits([case for _, case in files])
    stamped: list[Path] = []
    for path, case in files:
        if case.split is not None:
            continue
        text = path.read_text(encoding="utf-8")
        separator = "" if text.endswith("\n") else "\n"
        path.write_text(f"{text}{separator}split: {dealt[case.case_id].value}\n", encoding="utf-8")
        stamped.append(path)
    return tuple(stamped)


def _read(
    cases_root: Path, registry: Mapping[str, AgentSite[Any, Any]] | None
) -> tuple[tuple[Path, Case], ...]:
    """Every coder case beside the file it came from, in case-id order, or a loud refusal.

    The one reader in this module: the provider projects the cases out of it, and the stamper needs
    the paths beside them, so neither walks the tree itself and there is no second place a bucket
    could be skipped.
    """
    index = SITES if registry is None else registry
    site(CODER_SITE_ID, index)
    root = Path(cases_root) / CODER_SITE_ID
    if not root.is_dir():
        raise MissingCorpus(
            f"no case directory for site {CODER_SITE_ID!r} at {root} — a benchmark over an "
            "absent corpus is a number nobody can trust"
        )
    _refuse_strays(root)
    declared = frozenset(index)
    found = [
        (path, _case(path, bucket, declared))
        for bucket in Bucket
        for path in _bucket_files(root / bucket.value)
    ]
    return tuple(sorted(found, key=lambda pair: pair[1].case_id))


def _bucket_files(directory: Path) -> tuple[Path, ...]:
    """One bucket's case files in name order — nothing at all when the bucket has not been started.

    A missing bucket directory is not a defect: ``field/`` and ``replay/`` are gitignored, so a
    fresh clone simply has none, and refusing there would make the shipped corpus unloadable.
    """
    if not directory.is_dir():
        return ()
    _refuse_near_miss(directory)
    return tuple(sorted(directory.glob(f"*{CASE_SUFFIX}")))


def _refuse_strays(root: Path) -> None:
    """Refuse anything in the site directory that is not one of the four bucket directories.

    Both shapes are the same failure — a case nobody measures. A file beside the buckets is never
    read (the loader only walks buckets), and a directory the vocabulary does not name is a bucket
    somebody invented, whose cases would be invisible to every benchmark run over this corpus.
    """
    _refuse_near_miss(root)
    loose = next(iter(sorted(root.glob(f"*{CASE_SUFFIX}"))), None)
    if loose is not None:
        buckets = ", ".join(bucket.value for bucket in Bucket)
        raise MalformedCase(
            f"{loose}: a coder case lives in one of the bucket directories ({buckets}), not beside "
            "them — a file here is never read, and a case nobody measures is worse than none"
        )
    declared = {bucket.value for bucket in Bucket}
    unknown = sorted(
        path.name for path in root.iterdir() if path.is_dir() and path.name not in declared
    )
    if unknown:
        raise MalformedCase(
            f"{root}: sub-directory/directories {', '.join(repr(name) for name in unknown)} "
            f"is/are not a declared bucket — a coder corpus holds {', '.join(sorted(declared))}"
        )


def _refuse_near_miss(directory: Path) -> None:
    """Refuse a ``.yml`` near-miss: silently unread is how a case stops being measured."""
    stray = next(iter(sorted(directory.glob("*.yml"))), None)
    if stray is not None:
        raise MalformedCase(
            f"{stray}: a case file is named '<case id>{CASE_SUFFIX}' — rename it rather than "
            "leave it unread"
        )


def _case(path: Path, bucket: Bucket, declared: frozenset[str]) -> Case:
    """One file as a validated case, refused by path on bad YAML, a bad case, or a wrong bucket.

    The coder schema validates first, because its document-level entry point reports the generic
    *and* coder defects of a file in one refusal; :func:`~noctis.eval.case.parse_case` then reads
    the same document into the :class:`~noctis.eval.case.Case` a corpus is built from, and cannot
    fail once the combined pass has admitted it.
    """
    source = str(path)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise MalformedCase(f"{source}: not valid YAML — {error}") from error
    payload = parse_coder_case(
        document, case_id=path.stem, source=source, declared_site_ids=declared
    )
    if payload.bucket is not bucket:
        raise MalformedCase(
            f"{source}: declares {BUCKET_TAG_PREFIX}{payload.bucket.value} but sits in the "
            f"{bucket.value!r} bucket directory — a case is filed under what it is for, and a "
            "benchmark reports the tag"
        )
    case = parse_case(document, case_id=path.stem, source=source, declared_site_ids=declared)
    if case.site_id != CODER_SITE_ID:
        raise MalformedCase(
            f"{source}: site_id {case.site_id!r} does not match its directory "
            f"{CODER_SITE_ID!r} — a case lives in the directory of the site it exercises"
        )
    return case
