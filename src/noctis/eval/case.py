"""``Case`` — one benchmark ask, declared as data, and the strict validation that admits it.

A case is the unit a benchmark measures over: one thing a site is asked to do, small enough to
review in a pull request and stable enough to still mean the same thing a year later. This module
is the whole vocabulary — the frozen record, the two provenance forms, the split a corpus freezes,
the refusal a bad file earns — plus :class:`CaseProvider`, the one-method seam a corpus is loaded
through. Nothing here touches a directory; the YAML filesystem provider
(:mod:`noctis.eval.case_provider`) is the only thing that reads, which is what keeps this module
snapshot-cheap to test, exactly as the run record's pure builder is.

**A case carries no expected output, and the schema refuses one by name.** The expectation is the
site's own oracle — the coder's fresh-subprocess write gate, a scorer over a typed emit — not a
reference answer parked in a corpus file. A stored answer would be correct only until production
improved, and a benchmark whose corpus rots into disagreement with the thing it measures reports
regressions that are really staleness. So :data:`REFUSED_KEYS` names the shapes that mistake takes
(``expected``, ``answer``, ``solution``, ``output``, …) and refuses the file *by that key*, because
a silently ignored key is a curator who believes something is being checked that is not.

**Validation is strict, loud, and one-pass.** :func:`parse_case` reports every defect in a file in
a single :class:`MalformedCase` naming the file and each problem — the same posture as
:meth:`~noctis.eval.knobs.SiteKnobs.validate_overrides` and the settings overlay's refusals, so a
broken case is one edit rather than a fix-one-rerun loop. There is no lenient mode and no default
that papers over an absence: a case that cannot be read as written never reaches a benchmark
number, because a number computed over a corpus somebody half-understood is worse than no number.

**Provenance is exactly two forms.** ``mined:<run_id>`` (harvested from a real run, keyed by the id
that run minted — matched against :data:`~noctis.observability.runid.RUN_ID_RE` itself rather
than a lookalike pattern re-derived here) or ``authored:<YYYY-MM-DD>``. Between them a corpus can
always answer where it came from and how old it is, which is the question asked the moment a
benchmark number looks surprising.

**The split is a field here, and a decision elsewhere.** :class:`Split` is carried on the case and
validated when a file declares one, but *computing* the stratified tuning/holdout assignment is the
corpus's job (#197). So the field is ``None``-able and one-way: a file that declares no split loads
unassigned, :meth:`Case.assigned_to` returns a *new* case with the split stamped, and a case that
already carries one refuses reassignment. That refusal is the contamination discipline the
promotion gates enforce one layer down, moved up to the eval layer: a holdout case that could
quietly become a tuning case is a holdout nobody can trust.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from datetime import date
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from noctis.observability.runid import RUN_ID_RE

__all__ = [
    "CASE_KEYS",
    "REFUSED_KEYS",
    "REQUIRED_KEYS",
    "Case",
    "CaseProvider",
    "MalformedCase",
    "Provenance",
    "ProvenanceKind",
    "Split",
    "parse_case",
]


class Split(Enum):
    """Which half of the corpus a case belongs to, frozen once at corpus construction.

    ``TUNING`` is what a prompt or harness change may be justified on; ``HOLDOUT`` is what that
    justification is confirmed against, and is never looked at while iterating.
    """

    TUNING = "tuning"
    HOLDOUT = "holdout"


class ProvenanceKind(Enum):
    """Where a case came from: harvested from a run, or written by a person on a date."""

    MINED = "mined"
    AUTHORED = "authored"


class MalformedCase(ValueError):
    """A case document that cannot be admitted as written. One type every caller catches.

    The message names the file (or whatever label the caller passed as ``source``) and every
    defect found in it, so one read of one error is enough to fix the file.
    """


@dataclass(frozen=True)
class Provenance:
    """A case's origin, in one of the two forms a corpus file may spell it.

    ``reference`` is the run id for a mined case and the ISO date for an authored one, kept as the
    text the file carried so ``str(provenance)`` round-trips exactly what a curator wrote.
    """

    kind: ProvenanceKind
    reference: str

    def __str__(self) -> str:
        """The text form a case file carries — ``mined:<run_id>`` or ``authored:<YYYY-MM-DD>``."""
        return f"{self.kind.value}:{self.reference}"

    @property
    def mined_from(self) -> str | None:
        """The run id this case was harvested from, or ``None`` for an authored case."""
        return self.reference if self.kind is ProvenanceKind.MINED else None

    @property
    def authored_on(self) -> date | None:
        """The date this case was written, or ``None`` for a mined one."""
        return date.fromisoformat(self.reference) if self.kind is ProvenanceKind.AUTHORED else None

    @classmethod
    def parse(cls, text: object) -> Provenance:
        """Read one provenance string, refusing anything outside the two declared forms.

        Strict on both halves: the mined form's reference must be a real run id (the minter's own
        pattern), and the authored form's must be a dashed ISO date — ``authored:20260720`` and
        ``authored:2026-13-01`` are both refused, because a date a reader has to guess at is not an
        answer to "how old is this corpus".
        """
        if isinstance(text, str):
            kind, separator, reference = text.partition(":")
            if separator and kind == ProvenanceKind.MINED.value and RUN_ID_RE.match(reference):
                return cls(ProvenanceKind.MINED, reference)
            if separator and kind == ProvenanceKind.AUTHORED.value and _is_iso_date(reference):
                return cls(ProvenanceKind.AUTHORED, reference)
        raise MalformedCase(
            f"provenance must be 'mined:<run_id>' or 'authored:<YYYY-MM-DD>', got {text!r}"
        )


@dataclass(frozen=True)
class Case:
    """One benchmark ask: which site, what it is given, how it is labelled, where it came from.

    ``case_id`` is the corpus-unique name a record and a per-case artifact directory are keyed by
    (the YAML provider takes it from the file name, so the directory itself makes it unique);
    ``payload`` is the site's input, deep-frozen on construction so no consumer can edit the ask
    between two runs that claim to be the same benchmark; ``tags`` and ``difficulty`` are the
    labels a corpus stratifies and reports over; ``split`` is ``None`` until a corpus freezes it.
    """

    case_id: str
    site_id: str
    payload: Mapping[str, Any]
    provenance: Provenance
    tags: tuple[str, ...] = ()
    difficulty: Mapping[str, str] = field(default_factory=dict)
    split: Split | None = None

    def __post_init__(self) -> None:
        """Take the ask by value and read-only, all the way down: a case is a record of one ask."""
        object.__setattr__(self, "payload", _frozen(dict(self.payload)))
        object.__setattr__(self, "difficulty", MappingProxyType(dict(self.difficulty)))
        object.__setattr__(self, "tags", tuple(self.tags))

    def assigned_to(self, split: Split) -> Case:
        """This case with ``split`` stamped on it — the seam a corpus freezes its split through.

        Refuses a case that already carries a split: an assignment that could be recomputed is an
        assignment that cannot be trusted to mean the same thing tomorrow.
        """
        if self.split is not None:
            raise ValueError(
                f"case {self.case_id!r} is already assigned to {self.split.value} — a frozen "
                "split is never reassigned"
            )
        return replace(self, split=split)


# Every key a case file may declare, in the order a reader meets them. Anything else is refused.
CASE_KEYS: tuple[str, ...] = tuple(f.name for f in fields(Case) if f.name != "case_id")

# The three a file must declare. Tags, difficulty axes and the split are honestly optional: an
# unlabelled case is still a case, and an unassigned one is a case a corpus has not split yet.
REQUIRED_KEYS: tuple[str, ...] = ("site_id", "payload", "provenance")

# The keys that would smuggle a reference answer into a corpus file, refused by name. The list is
# deliberately generous: every one of these is a curator expecting an expectation to be checked.
REFUSED_KEYS: frozenset[str] = frozenset(
    {
        "answer",
        "correct",
        "correct_answer",
        "expectation",
        "expected",
        "expected_outcome",
        "expected_output",
        "gold",
        "gold_output",
        "golden",
        "ground_truth",
        "oracle",
        "output",
        "reference_output",
        "solution",
        "truth",
    }
)

_NO_EXPECTATION = (
    "a case carries no expected output; the site's oracle is the expectation, so a corpus "
    "file never rots into disagreement with it"
)


def parse_case(
    document: object,
    *,
    case_id: str,
    source: str,
    declared_site_ids: Collection[str] | None = None,
) -> Case:
    """Read one case document into a :class:`Case`, or refuse it naming every defect at once.

    ``source`` is what the refusal names — a file path for the YAML provider, any honest label for
    another provider. ``declared_site_ids`` cross-checks the case's site against a set of declared
    ids (the provider passes the site registry's); left ``None``, the site id is taken as written,
    because this module knows nothing about which sites exist and inventing that coupling would
    make the pure validator un-testable without the shipped declarations.
    """
    if not isinstance(document, Mapping):
        raise MalformedCase(
            f"{source}: a case is a mapping of {', '.join(CASE_KEYS)} — "
            f"got {type(document).__name__}"
        )

    defects: list[str] = []
    if not case_id.strip():
        defects.append("a case id must be a non-empty name")
    defects.extend(_key_defects(document))

    site_id = ""
    if "site_id" in document:
        candidate = document["site_id"]
        if not isinstance(candidate, str) or not candidate.strip():
            defects.append(f"site_id must be a non-empty string, got {candidate!r}")
        elif declared_site_ids is not None and candidate not in declared_site_ids:
            declared = ", ".join(sorted(declared_site_ids)) or "no sites"
            defects.append(
                f"site_id {candidate!r} is not a declared site — declared sites: {declared}"
            )
        else:
            site_id = candidate

    payload: Mapping[str, Any] = {}
    if "payload" in document:
        candidate = document["payload"]
        if not isinstance(candidate, Mapping) or not candidate:
            defects.append(
                f"payload must be a non-empty mapping of the site's input, got {candidate!r}"
            )
        elif not all(isinstance(key, str) for key in candidate):
            defects.append(f"payload keys must be strings, got {sorted(map(repr, candidate))}")
        else:
            payload = candidate

    provenance: Provenance | None = None
    if "provenance" in document:
        try:
            provenance = Provenance.parse(document["provenance"])
        except MalformedCase as refusal:
            defects.append(str(refusal))

    tags, tag_defect = _tags(document.get("tags", ()))
    difficulty, difficulty_defect = _difficulty(document.get("difficulty", {}))
    split, split_defect = _split(document.get("split"))
    defects.extend(defect for defect in (tag_defect, difficulty_defect, split_defect) if defect)

    if defects:
        raise MalformedCase(f"{source}: {'; '.join(defects)}")
    if provenance is None:  # only reachable if a defect above went unreported
        raise MalformedCase(f"{source}: missing required key 'provenance'")
    return Case(
        case_id=case_id,
        site_id=site_id,
        payload=payload,
        provenance=provenance,
        tags=tags,
        difficulty=difficulty,
        split=split,
    )


@runtime_checkable
class CaseProvider(Protocol):
    """Where a benchmark's cases come from — one method, so a new corpus shape is a new class.

    The v1 implementation is :class:`~noctis.eval.case_provider.YamlCaseProvider` over per-site
    directories of YAML files, and files stay the reviewable interchange format. The protocol
    exists so the *consumers* — runner, metrics, record — never learn the directory layout, and a
    replay database, a mined capture or a generated corpus can arrive later as another provider
    rather than as an edit to all three.
    """

    def load(self, site_id: str) -> tuple[Case, ...]:
        """Every case declared for ``site_id``, in a stable order, or a loud refusal."""
        ...


def _key_defects(document: Mapping[Any, Any]) -> list[str]:
    """The refused, unknown and missing keys of a document, as messages naming each one."""
    defects = [f"key {key!r} is refused — {_NO_EXPECTATION}" for key in sorted(_refused(document))]
    unknown = sorted(str(key) for key in document if str(key) not in CASE_KEYS)
    unknown = [key for key in unknown if key not in REFUSED_KEYS]
    if unknown:
        defects.append(
            f"unknown key(s) {', '.join(repr(key) for key in unknown)} — "
            f"a case declares {', '.join(CASE_KEYS)}"
        )
    missing = [key for key in REQUIRED_KEYS if key not in document]
    if missing:
        defects.append(f"missing required key(s) {', '.join(repr(key) for key in missing)}")
    return defects


def _refused(document: Mapping[Any, Any]) -> list[str]:
    """The expected-output-shaped keys a document carries, matched case-insensitively."""
    return [str(key) for key in document if str(key).lower() in REFUSED_KEYS]


def _tags(candidate: object) -> tuple[tuple[str, ...], str | None]:
    """A document's tags as a tuple, or the defect that stops them being one."""
    if isinstance(candidate, str) or not isinstance(candidate, Sequence):
        return (), f"tags must be a list of distinct non-empty strings, got {candidate!r}"
    if not all(isinstance(tag, str) and tag.strip() for tag in candidate):
        return (), f"tags must be a list of distinct non-empty strings, got {candidate!r}"
    tags = tuple(candidate)
    repeated = sorted({tag for tag in tags if tags.count(tag) > 1})
    if repeated:
        return (), f"tags repeat {', '.join(repr(tag) for tag in repeated)}"
    return tags, None


def _difficulty(candidate: object) -> tuple[Mapping[str, str], str | None]:
    """A document's difficulty axes as an axis→level mapping, or the defect refusing them.

    Levels are strings because they are stratification buckets, not measurements: a corpus splits
    on ``{"reasoning": "hard"}``, and a number there would invite arithmetic on a label.
    """
    defect = (
        f"difficulty must be a mapping of axis name to a non-empty string level, got {candidate!r}"
    )
    if not isinstance(candidate, Mapping):
        return {}, defect
    for axis, level in candidate.items():
        if not isinstance(axis, str) or not isinstance(level, str) or not level.strip():
            return {}, defect
    return dict(candidate), None


def _split(candidate: object) -> tuple[Split | None, str | None]:
    """A document's split as a :class:`Split`, ``None`` when it declares none, or the defect."""
    if candidate is None:
        return None, None
    try:
        return Split(candidate), None
    except ValueError:
        return None, f"split must be 'tuning' or 'holdout', got {candidate!r}"


def _is_iso_date(reference: str) -> bool:
    """Whether a string is a dashed ISO calendar date — the one authored-provenance shape."""
    if len(reference) != 10 or reference[4] != "-" or reference[7] != "-":
        return False
    try:
        date.fromisoformat(reference)
    except ValueError:
        return False
    return True


def _frozen(value: Any) -> Any:
    """A payload value with every mapping and sequence in it made read-only, all the way down."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_frozen(item) for item in value)
    return value
