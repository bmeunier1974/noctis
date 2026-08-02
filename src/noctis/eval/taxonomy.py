"""The failure-taxonomy mechanism — per-site classifiers over gate errors, with a catch-all.

"The coder failed 40% of the time" is a number nobody can act on. *How* it failed is: a run whose
failures are three-quarters "the emitted file would not import" is a prompt problem, and one whose
failures are three-quarters "the strategy traded twice in ten years" is a thesis problem. Counting
those apart needs a vocabulary of failure classes and a rule for putting an error string in one of
them, which is what this module is — and *only* what it is. No site's vocabulary lives here: the
coder's classes are the coder epic's to write, DECIDE's are DECIDE's. This is the shape they are
all held to.

**Per-site by construction, because taxonomies do not generalize.** A vocabulary is registered
against one site id and is reachable only through it. Two sites may both call a class ``refused``
and mean two unrelated things; neither site's matchers are ever consulted for the other's errors.
That is not a convenience — a shared taxonomy would quietly make one site's rot look like another's
health. What crosses the boundary into shared reporting is a :class:`FailureBreakdown`: class names,
counts and shares, with no site id and no matcher in it, so a report can aggregate without ever
being in a position to mix two vocabularies up.

**The catch-all is the honesty gauge.** Every vocabulary is closed, so every vocabulary is wrong
eventually — a new gate message arrives, and the classes that used to cover the failures stop
covering them. :data:`UNCLASSIFIED` is the reserved class every unmatched error lands in, and it is
reported like any other class, at zero when nothing missed. Its *share rising over successive
benchmark runs* is the signal that the classifiers have rotted, which is why it may never be
silently dropped from a breakdown and why no site may register a class under its name.

**Overlap is decided, not accidental: first registered wins.** Matchers are predicates written by
hand, so two of them will eventually both recognise one error. Rather than refuse the ambiguity
(which would make a vocabulary un-extendable) or resolve it by dictionary order (which would make
the answer depend on a name), the classes are tried in registration order and the first match is
the answer. Registration order is therefore priority order, and a vocabulary reads specific-first.

**Purity, and no global.** Classification is a pure function of the strings it is handed: no clock,
no file, no config, no randomness — the same discipline ``reporting/run_record`` and
``observability/metrics`` are held to, and the reason a benchmark record can be re-derived from its
inputs. A :class:`FailureTaxonomy` is an instance a caller constructs, never a module-level
singleton populated by whatever happened to be imported — the same posture as the strategy-family
registry, which every session builds its own of.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    "UNCLASSIFIED",
    "FailureBreakdown",
    "FailureClass",
    "FailureTaxonomy",
    "Matcher",
    "UnknownVocabulary",
    "VocabularyError",
]

#: The reserved class every error no matcher recognises lands in. Always present in a breakdown.
UNCLASSIFIED = "unclassified"

# One failure class's whole behaviour: does this error string belong to me? Pure, total, and cheap
# — it is called once per class per error, and a matcher that reads a file would make a benchmark
# record depend on the machine that scored it.
Matcher = Callable[[str], bool]


class VocabularyError(ValueError):
    """A vocabulary that cannot mean what it says: a duplicate, the reserved name, or nothing.

    Raised at registration, where it costs one edit, rather than at classification, where it would
    cost a benchmark run and a record whose class counts nobody can trust.
    """


class UnknownVocabulary(KeyError):
    """Classification asked of a site that registered no vocabulary. Names what is registered."""

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


@dataclass(frozen=True)
class FailureClass:
    """One named class of failure and the pure predicate that recognises it.

    ``name`` is what shared reporting sees; ``matches`` is never seen outside the taxonomy.
    """

    name: str
    matches: Matcher


@dataclass(frozen=True)
class FailureBreakdown:
    """One batch of errors, counted by class — the whole of what crosses into shared reporting.

    ``counts`` and ``shares`` are read-only mappings keyed by class name in vocabulary order with
    :data:`UNCLASSIFIED` last, and every class of the vocabulary appears, at zero when it never
    matched. ``shares`` are fractions of ``total``; an empty batch is all zeros rather than a
    division nobody can perform.
    """

    counts: Mapping[str, int]
    shares: Mapping[str, float]
    total: int

    @property
    def unclassified_share(self) -> float:
        """The rot gauge, readable on its own: the fraction no matcher recognised."""
        return self.shares[UNCLASSIFIED]


class FailureTaxonomy:
    """Per-site failure vocabularies, and classification against one of them.

    Construct one per benchmark session and register each site's classes into it. Nothing is shared
    between instances, and nothing is shared between sites within an instance.
    """

    def __init__(self) -> None:
        self._vocabularies: dict[str, tuple[FailureClass, ...]] = {}

    def register(self, site_id: str, classes: Iterable[FailureClass]) -> None:
        """Register one site's closed vocabulary, in priority (first-match-wins) order.

        Refuses, naming the site and the offence: a class name declared twice, a class taking the
        reserved :data:`UNCLASSIFIED` name, an empty vocabulary (which would classify nothing and
        is indistinguishable from the site nobody has written a taxonomy for), and a second
        vocabulary for a site that already has one — closed means closed, and re-registering would
        silently re-label every record scored before the change.
        """
        if site_id in self._vocabularies:
            raise VocabularyError(
                f"site {site_id!r} already registered a failure vocabulary — a vocabulary is "
                f"closed; declare every class in one registration"
            )
        declared = tuple(classes)
        if not declared:
            raise VocabularyError(
                f"site {site_id!r} registered a failure vocabulary with no classes — a vocabulary "
                f"that classifies nothing says nothing that {UNCLASSIFIED!r} does not"
            )
        seen: set[str] = set()
        for failure_class in declared:
            if failure_class.name == UNCLASSIFIED:
                raise VocabularyError(
                    f"site {site_id!r} declares a class named {UNCLASSIFIED!r} — that name is "
                    f"reserved for the catch-all every unmatched error lands in"
                )
            if failure_class.name in seen:
                raise VocabularyError(
                    f"site {site_id!r} declares the class {failure_class.name!r} twice — class "
                    f"names are unique within a site's vocabulary"
                )
            seen.add(failure_class.name)
        self._vocabularies[site_id] = declared

    def registered_sites(self) -> tuple[str, ...]:
        """Every site with a registered vocabulary, in registration order."""
        return tuple(self._vocabularies)

    def class_names(self, site_id: str) -> tuple[str, ...]:
        """One site's classes in registration order, with the catch-all last."""
        return tuple(failure_class.name for failure_class in self._classes(site_id)) + (
            UNCLASSIFIED,
        )

    def classify_one(self, site_id: str, error: str) -> str:
        """The name of the first class of ``site_id`` whose matcher recognises ``error``.

        :data:`UNCLASSIFIED` when none does — never an exception, because an error nobody wrote a
        matcher for is a fact to count, not a failure to classify.
        """
        for failure_class in self._classes(site_id):
            if failure_class.matches(error):
                return failure_class.name
        return UNCLASSIFIED

    def classify(self, site_id: str, errors: Iterable[str]) -> FailureBreakdown:
        """Count a batch of error strings into ``site_id``'s classes.

        Order-independent: each error is classified on its own, so a permuted batch is the same
        breakdown. Every class is reported, the catch-all included.
        """
        names = self.class_names(site_id)
        counts = dict.fromkeys(names, 0)
        total = 0
        for error in errors:
            counts[self.classify_one(site_id, error)] += 1
            total += 1
        shares = {name: (counts[name] / total if total else 0.0) for name in names}
        return FailureBreakdown(
            counts=MappingProxyType(counts), shares=MappingProxyType(shares), total=total
        )

    def _classes(self, site_id: str) -> tuple[FailureClass, ...]:
        """One site's declared classes, or the refusal that names the sites that have one."""
        if site_id not in self._vocabularies:
            registered = ", ".join(self._vocabularies) or "no sites"
            raise UnknownVocabulary(
                f"site {site_id!r} registered no failure vocabulary — registered: {registered}"
            )
        return self._vocabularies[site_id]
