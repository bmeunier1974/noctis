"""``Corpus`` — a site's cases, split into tuning and holdout once, and hashed as one thing.

A benchmark's tuning half is what a prompt or harness change may be *justified* on; its holdout is
what that justification is *confirmed* against. That distinction only means something if the two
halves are decided before anybody looks at either and never re-decided afterwards — which is
promotion's forward/symbol-holdout discipline (rules 3 and 4) moved up to the eval layer. So this
module does exactly two things to a pile of cases a provider already loaded: it **deals the split
once, deterministically**, and it **hashes the result**. It is pure — no directory, no clock, no
seed, no configuration — for the same reason the run record's builder is: a number is only
reproducible if the thing it was computed over can be rebuilt from its inputs alone.

**The split rule, written out, because a rule nobody can restate is a rule nobody can check.**

1. **Stratify by the difficulty axes.** A case's stratum is its ``difficulty`` mapping rendered as
   sorted ``axis=level`` pairs, so identical axes are one stratum whatever order a file wrote them
   in, and cases declaring no axes are one stratum of their own. Tags are deliberately *not* part
   of it: tags are free-form labels, and stratifying on them would shatter a corpus into strata of
   one — every one of which lands in tuning (see the rounding below), leaving no holdout at all.
2. **Order within a stratum by the hash of the case id** — ``sha256(case_id)``, tie-broken by the
   id itself. Not by file order (which a rename changes), not by insertion order (which two
   providers disagree about), and never by :func:`hash`, ``set`` iteration or a seeded shuffle,
   all three of which make the same corpus split differently in the next process.
3. **Cut at the holdout share.** A stratum of *n* holds ``round-half-up(n × 3/10)`` holdout cases —
   the *last* that many in the hash order — and the rest tuning. The arithmetic is exact
   (:class:`~fractions.Fraction`, never a float, whose binary rounding lands on the wrong side of a
   ``.5`` boundary for some sizes). Small strata round toward tuning: **1 → 0 holdout**, 2 → 1,
   3 → 1, 5 → 2. A single-case stratum going wholly to tuning is the honest reading — a holdout of
   one case out of one leaves nothing to iterate on, and a corpus that small is a corpus somebody
   has started rather than a benchmark.
4. **A case that already carries a split keeps it, and counts toward its stratum's quota.** Frozen
   is frozen (:meth:`~noctis.eval.case.Case.assigned_to` refuses reassignment), so re-loading an
   assigned corpus re-deals nothing; and because the quota counts what is already frozen, pinning
   three cases of a ten-case stratum to the holdout leaves the other seven in tuning rather than
   dealing a second holdout on top. A stratum whose frozen cases already exceed its quota keeps
   them — the corpus reports what it is, it does not un-freeze anything to hit a ratio.

The consequence worth stating: the assignment depends on the *whole* stratum, so adding a case can
move the unassigned ones. That is the price of hitting the ratio exactly at small sizes, and it is
paid where it belongs — a corpus that has stopped changing is a corpus whose split has stopped
moving, and a case that must never move is a case a file pins with ``split:``.

**The digest.** ``sha256`` over a canonical rendering of the corpus — a scheme tag, the site id,
the case count, then one canonical JSON line per case (sorted keys, the payload's nested mappings
sorted too) sorted among themselves. Sorting the lines is what makes it order-independent across
load orders while staying sensitive to any content change: edit a payload, a tag, an axis, a
provenance or a case id and the digest moves. The rendering is taken *after* the split is frozen,
so the digest states both what was asked and how it was divided — two runs over the same cases
partitioned differently are not comparable, and an identity that could not tell them apart would
say they were. :class:`CorpusIdentity` is the triple a record folds into its ``comparable_key``,
rendered ``coder|<digest>|<count>`` exactly as
:class:`~noctis.observability.engine_id.ComparableKey` renders the engine's.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, NamedTuple

from noctis.eval.case import Case, Split

__all__ = ["HOLDOUT_SHARE", "Corpus", "CorpusError", "CorpusIdentity"]

#: The holdout's share of every stratum, as an exact ratio — never a float, whose binary rounding
#: puts some stratum sizes on the wrong side of the half-way cut.
HOLDOUT_SHARE = Fraction(3, 10)

# The digest's rendering scheme, hashed in as a prefix. A change to *how* a corpus is rendered
# (not to what it contains) is a new scheme name, so an old digest is never silently re-meant.
_DIGEST_SCHEME = "corpus/1"

# Sixteen hex characters, the length the engine and prompt fingerprints already use: short enough
# to read in a record, long enough that two corpora colliding is not a thing anybody plans for.
_DIGEST_CHARS = 16


class CorpusError(ValueError):
    """A set of cases that cannot be one corpus: a repeated id, or a case from another site."""


class CorpusIdentity(NamedTuple):
    """What a benchmark record folds in to say *which* corpus a number was computed over.

    The digest carries the strict guarantee (content **and** partition); ``case_count`` is beside
    it because a reader comparing two records wants to see a corpus grow without diffing hashes.
    """

    site_id: str
    corpus_digest: str
    case_count: int

    def __str__(self) -> str:
        """``coder|4f1c0c2f9b7a5d18|42`` — one greppable label for a corpus."""
        return "|".join(str(part) for part in self)


@dataclass(frozen=True)
class Corpus:
    """One site's cases, held in case-id order with a tuning/holdout split frozen on each.

    Constructing a corpus *is* the deal: the split is computed in ``__post_init__`` and stamped
    through :meth:`~noctis.eval.case.Case.assigned_to`, so there is no window in which a corpus
    exists un-split and no second entry point that could deal it differently.
    """

    site_id: str
    cases: tuple[Case, ...] = ()

    def __post_init__(self) -> None:
        """Refuse a corpus that cannot be one, then freeze the split, once."""
        loaded = tuple(self.cases)
        _refuse_repeats(loaded)
        _refuse_foreign(self.site_id, loaded)
        object.__setattr__(self, "cases", _dealt(loaded))

    def __len__(self) -> int:
        """How many cases this corpus holds."""
        return len(self.cases)

    @property
    def tuning(self) -> tuple[Case, ...]:
        """The half a prompt or harness change may be justified on, in case-id order."""
        return self._half(Split.TUNING)

    @property
    def holdout(self) -> tuple[Case, ...]:
        """The half that justification is confirmed against, in case-id order."""
        return self._half(Split.HOLDOUT)

    @property
    def digest(self) -> str:
        """This corpus's content hash — the same corpus, the same 16 hex characters, always."""
        body = "\n".join(
            (
                _DIGEST_SCHEME,
                self.site_id,
                str(len(self.cases)),
                *sorted(_rendered(case) for case in self.cases),
            )
        )
        return hashlib.sha256(body.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]

    @property
    def identity(self) -> CorpusIdentity:
        """The site, digest and case count a record carries beside the engine's own identity."""
        return CorpusIdentity(
            site_id=self.site_id, corpus_digest=self.digest, case_count=len(self.cases)
        )

    def _half(self, split: Split) -> tuple[Case, ...]:
        return tuple(case for case in self.cases if case.split is split)


def _refuse_repeats(cases: Sequence[Case]) -> None:
    """Refuse a corpus whose case ids are not unique — identity is keyed by them."""
    counted: dict[str, int] = {}
    for case in cases:
        counted[case.case_id] = counted.get(case.case_id, 0) + 1
    repeated = sorted(case_id for case_id, count in counted.items() if count > 1)
    if repeated:
        raise CorpusError(
            f"case id(s) {', '.join(repr(case_id) for case_id in repeated)} appear more than "
            "once — a corpus is keyed by case id, and two cases sharing one are a result nobody "
            "can attribute"
        )


def _refuse_foreign(site_id: str, cases: Iterable[Case]) -> None:
    """Refuse a case belonging to another site — a corpus measures exactly one of them."""
    foreign = sorted({case.site_id for case in cases if case.site_id != site_id})
    if foreign:
        raise CorpusError(
            f"corpus for site {site_id!r} holds case(s) declaring site(s) "
            f"{', '.join(repr(other) for other in foreign)} — every case of a corpus exercises "
            "the site the corpus measures"
        )


def _dealt(cases: Iterable[Case]) -> tuple[Case, ...]:
    """Every case with a split frozen on it, in case-id order: the whole deal, in one pass."""
    strata: dict[str, list[Case]] = {}
    for case in sorted(cases, key=lambda case: case.case_id):
        strata.setdefault(_stratum(case), []).append(case)
    assigned = [case for members in strata.values() for case in _deal(members)]
    return tuple(sorted(assigned, key=lambda case: case.case_id))


def _stratum(case: Case) -> str:
    """A case's stratum: its difficulty axes, canonically rendered. No axes is a stratum too."""
    return "|".join(f"{axis}={level}" for axis, level in sorted(case.difficulty.items()))


def _deal(members: Sequence[Case]) -> list[Case]:
    """One stratum's cases, split at the holdout share with the frozen ones counted in.

    The already-assigned cases pass through untouched; the rest are ordered by the hash of their
    id and cut so that the stratum *as a whole* lands on the share. When the frozen cases already
    fill (or overfill) the quota there is nothing left to deal to the holdout, and when they leave
    a bigger quota than there are unassigned cases every unassigned case goes to the holdout — in
    both directions the corpus reports what it is rather than un-freezing something to hit a ratio.
    """
    frozen = [case for case in members if case.split is not None]
    unassigned = sorted((case for case in members if case.split is None), key=_hash_order)
    quota = int(len(members) * HOLDOUT_SHARE + Fraction(1, 2))
    already = sum(1 for case in frozen if case.split is Split.HOLDOUT)
    wanted = max(0, quota - already)
    cut = max(0, len(unassigned) - wanted)
    return [
        *frozen,
        *(case.assigned_to(Split.TUNING) for case in unassigned[:cut]),
        *(case.assigned_to(Split.HOLDOUT) for case in unassigned[cut:]),
    ]


def _hash_order(case: Case) -> tuple[str, str]:
    """The deal order within a stratum: the case id's digest, tie-broken by the id itself.

    A content hash rather than the id's own order, so a corpus whose ids share a prefix
    (``case-01``, ``case-02``, …) does not put its whole tail in the holdout; and a *stable* hash
    rather than :func:`hash`, so the deal survives a new process.
    """
    return hashlib.sha256(case.case_id.encode("utf-8")).hexdigest(), case.case_id


def _rendered(case: Case) -> str:
    """One case as a canonical JSON line — the unit the corpus digest is taken over."""
    return json.dumps(
        {
            "case_id": case.case_id,
            "site_id": case.site_id,
            "payload": _plain(case.payload),
            "provenance": str(case.provenance),
            "tags": list(case.tags),
            "difficulty": _plain(case.difficulty),
            "split": None if case.split is None else case.split.value,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _plain(value: Any) -> Any:
    """A frozen payload value as plain JSON-renderable data: read-only maps and tuples undone."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value
