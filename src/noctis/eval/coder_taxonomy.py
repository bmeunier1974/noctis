"""The coder's failure vocabulary — ten classes over the write gate's own wording.

"The coder failed 40% of the time" is a number nobody can act on; "half of those failures were the
file not importing" is. This module is that second number's vocabulary: the closed set of ways a
coder attempt dies, each class annotated with the **knob** its share points at, registered into the
:mod:`noctis.eval.taxonomy` mechanism as the site ``coder``.

**Nine recognizers plus the escape hatch — and the escape hatch is the mechanism's.** The epic names
ten classes and calls the tenth ``other`` ("unmatched — grow the taxonomy"); the mechanism reserves
:data:`~noctis.eval.taxonomy.UNCLASSIFIED` for exactly that bucket and refuses to let a site declare
a class under it. Inventing a *second* always-empty ``other`` class beside it would put two names on
one idea and report eleven classes for a ten-class vocabulary. So the reconciliation is by identity:
:data:`OTHER` **is** ``UNCLASSIFIED``. :data:`CODER_CLASSES` declares all ten rows (so the knob
table is complete and ``len`` is honestly ten), the last being the escape hatch — never registered,
never consulted (its matcher is total and always false), reached only the way the mechanism reaches
it: nothing else matched. Unmatched therefore never raises and never guesses, in the pure classifier
and in a registered taxonomy alike.

**The recognizers are the engine's, not copies of it.** A classifier that re-implements a gate
message drifts away from the gate in silence, and a class that quietly stops matching looks exactly
like a class of failure that stopped happening — the one confusion this module exists to prevent. So
the warmup class matches through :func:`~noctis.strategies.library.is_warmup_too_large` (the
engine's own predicate over its stable marker), and the API-misuse class searches with the *same
compiled patterns* the retry enricher searches with (:mod:`noctis.research.contract_sheet`). What
cannot be imported — messages the gate builds inline — is quoted with the module and function that
emits it, and pinned by a fixture in ``tests/test_eval_coder_taxonomy.py``.

**Registration order is precedence order** (the mechanism is first-match-wins), so the vocabulary
reads most-specific-first:

1. ``warmup_too_large`` — one stable marker, and it also wears the ``scenario '<name>':`` shape;
2. ``api_misuse`` — the two hint-matcher shapes, which arrive wrapped inside scenario and replay
   errors ("scenarios() raised TypeError: trend() got an unexpected keyword argument") and under a
   bare ``TypeError:`` head that the broad import match would otherwise swallow;
3. ``name_mismatch`` — two exact gate sentences;
4. ``signals_parity`` — one exact gate sentence;
5. ``structural_invariant`` — the Tier-1 invariants and the source lint, before the scenario class
   because every invariant failure is reported *on a tape* and shares its prefix;
6. ``scenario_violation`` — the declared/stamped tape not honoured, and the tape-shape contract;
7. ``truncated`` — before ``no_code_block``, because the output-limit correction also says no
   complete code block landed; the difference between them is the whole point of the pair;
8. ``no_code_block`` — a prose-only reply;
9. ``import_error`` — the broad match, last: an exception type crossing the gate boundary
   *unwrapped* means the module blew up before the gate could grade it, which is true of anything
   that reaches here without a shape above recognising it;
10. ``other`` — the escape hatch, whose share rising is the signal to grow this list.

Pure: classification is a function of the string handed in, with no clock, no file, no config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from noctis.eval.coder_distill_sites import CODER_SITE
from noctis.eval.taxonomy import UNCLASSIFIED, FailureClass, FailureTaxonomy, Matcher
from noctis.research.contract_sheet import _UNEXPECTED_KWARG_RE, _UPDATE_ARITY_RE
from noctis.strategies.library import StrategyValidationError, is_warmup_too_large

__all__ = [
    "CODER_CLASSES",
    "CODER_SITE_ID",
    "OTHER",
    "UNEXPECTED_KWARG_PATTERN",
    "UPDATE_ARITY_PATTERN",
    "CoderFailure",
    "classify_coder_failure",
    "coder_failure_classes",
    "knob_for",
    "register_coder_taxonomy",
]

#: The site this vocabulary belongs to, taken from the declaration itself so the two cannot drift.
CODER_SITE_ID: str = CODER_SITE.id

#: The epic's tenth class, ``other``, IS the mechanism's reserved catch-all — one bucket, one name.
OTHER: str = UNCLASSIFIED

# The retry enricher's two recognized API-misuse shapes, re-exported under public names rather than
# re-written: an attempt whose error the retry prompt can enrich with a true signature is, by
# definition, an attempt that misused the declared API. Same compiled objects, so a change to either
# shape moves the hint and this class together.
UPDATE_ARITY_PATTERN = _UPDATE_ARITY_RE
UNEXPECTED_KWARG_PATTERN = _UNEXPECTED_KWARG_RE

# A Tier-1 invariant failure and an unmet expectation are both reported against a named tape, so the
# prefix alone cannot tell them apart — which is why the invariant class is registered first.
_TAPE_PREFIX_RE = re.compile(r"scenario '[^']*':")

# An exception type at the head of the message, as the subprocess validator prints it
# ("{type(exc).__name__}: {exc}", library.py::_main), and as the gate then re-raises it.
_UNWRAPPED_EXCEPTION_RE = re.compile(r"\s*(?P<type>[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\b")

# The gate's own error type: its name at the head means the gate *graded* the file and refused it,
# so the message is one of the specific shapes above — never the broad import match.
_GATE_ERROR_NAME: str = StrategyValidationError.__name__

# Quoted markers, each with the emitting function. Imported where the engine exposes a constant
# (the warmup marker, the two API shapes); quoted where it builds the sentence inline.
_NAME_MISMATCH_MARKERS = (
    # library.py::_validate_file — "class sets name='x' but the strategy/file name is 'y'"
    "but the strategy/file name is",
    # library.py::load_and_register — "class name attribute 'X' != file name 'x'"
    "!= file name",
)

_SIGNALS_PARITY_MARKERS = (
    # library.py::_validate_file — the vectorised override against the replayed per-bar path
    "signals() disagrees with the on_bar replay",
)

_STRUCTURAL_MARKERS = (
    # strategies/scenarios Tier-1 invariants: warmup honesty, determinism, truncation, price scale
    "warmup dishonest",
    "non-deterministic replay",
    "signals() looks ahead",
    "price-scale dependent",
    # structure.py — the raw-source lint that runs before the file is even imported
    "duplicate definition of",
    "dead state ",
)

_SCENARIO_MARKERS = (
    # strategies/scenarios::check_scenario_contract — the tape-shape contract and its refusals
    "scenarios()",
    "must declare known-outcome scenarios",
    "at least one scenario must",
    "scenario names must be unique",
    "scenarios, got",
)

_TRUNCATED_MARKERS = (
    # author.py::_extraction_error, stop_reason == "length"
    "cut off by the output-token limit",
)

_NO_CODE_BLOCK_MARKERS = (
    # author.py::_extraction_error, any other stop reason — the reply was prose
    "carried no ```python code block",
    "no code block",
)

_IMPORT_MARKERS = (
    # library.py::_load_module — the loader could not even build a module from the file
    "cannot import",
    # library.py::_find_strategy_class — it imported, but yielded no single strategy class to grade
    "expected exactly one TraderStrategy subclass",
)

#: The knob an unmatched failure points at: this list, not a setting.
_OTHER_KNOB = "grow the taxonomy"


@dataclass(frozen=True)
class CoderFailure:
    """One class of coder failure: its name, the knob its share points at, and its recognizer.

    ``knob`` is the actionable half — what an operator reading a benchmark breakdown would reach for
    if this class dominated it — and it is carried beside the matcher so a class can never be added
    without someone saying what a rising share of it would mean.
    """

    name: str
    knob: str
    matches: Matcher

    def as_failure_class(self) -> FailureClass:
        """This row as the mechanism sees it — name and matcher, the knob left behind."""
        return FailureClass(name=self.name, matches=self.matches)


def _has_any(error: str, markers: tuple[str, ...]) -> bool:
    """Whether any quoted marker appears in ``error``. Total over every string, including empty."""
    return any(marker in error for marker in markers)


def _is_api_misuse(error: str) -> bool:
    """A State ``.update()`` arity mistake, or an unexpected keyword argument to a declared row."""
    return bool(UPDATE_ARITY_PATTERN.search(error) or UNEXPECTED_KWARG_PATTERN.search(error))


def _is_name_mismatch(error: str) -> bool:
    return _has_any(error, _NAME_MISMATCH_MARKERS)


def _is_signals_parity(error: str) -> bool:
    return _has_any(error, _SIGNALS_PARITY_MARKERS)


def _is_structural_invariant(error: str) -> bool:
    return _has_any(error, _STRUCTURAL_MARKERS)


def _is_scenario_violation(error: str) -> bool:
    """The stamped or declared tape was not honoured, or the tape-shape contract was refused."""
    return bool(_TAPE_PREFIX_RE.search(error)) or _has_any(error, _SCENARIO_MARKERS)


def _is_truncated(error: str) -> bool:
    return _has_any(error, _TRUNCATED_MARKERS)


def _is_no_code_block(error: str) -> bool:
    return _has_any(error, _NO_CODE_BLOCK_MARKERS)


def _is_import_error(error: str) -> bool:
    """The module never became a strategy class the gate could grade.

    Either the loader/discovery said so in the gate's words, or an exception type that is *not* the
    gate's own crossed the boundary unwrapped — which is what a file blowing up on import looks like
    by the point this broad match is reached.
    """
    if _has_any(error, _IMPORT_MARKERS):
        return True
    head = _UNWRAPPED_EXCEPTION_RE.match(error)
    return head is not None and head.group("type") != _GATE_ERROR_NAME


def _never(error: str) -> bool:
    """The escape hatch's matcher: it recognises nothing, because nothing recognising it is the
    definition of the class. Never registered and never consulted — the mechanism owns this bucket
    — but declared so every row of :data:`CODER_CLASSES` has the same shape."""
    return False


#: The closed vocabulary, in precedence (first-match-wins) order, escape hatch last.
CODER_CLASSES: tuple[CoderFailure, ...] = (
    CoderFailure(
        name="warmup_too_large",
        knob="fixed-oracle block wording, spec-path only",
        matches=is_warmup_too_large,
    ),
    CoderFailure(
        name="api_misuse",
        knob="contract sheet coverage, hint enrichment",
        matches=_is_api_misuse,
    ),
    CoderFailure(
        name="name_mismatch",
        knob="prompt contract rules",
        matches=_is_name_mismatch,
    ),
    CoderFailure(
        name="signals_parity",
        knob="model capability",
        matches=_is_signals_parity,
    ),
    CoderFailure(
        name="structural_invariant",
        knob="model capability — the serious one",
        matches=_is_structural_invariant,
    ),
    CoderFailure(
        name="scenario_violation",
        knob="feasibility rules, thinking dial, oracle mode",
        matches=_is_scenario_violation,
    ),
    CoderFailure(
        name="truncated",
        knob="coder max-tokens, thinking allowance, thinking dial",
        matches=_is_truncated,
    ),
    CoderFailure(
        name="no_code_block",
        knob="prompt output rules, model instruction-following",
        matches=_is_no_code_block,
    ),
    CoderFailure(
        name="import_error",
        knob="model capability / contract sheet",
        matches=_is_import_error,
    ),
    CoderFailure(name=OTHER, knob=_OTHER_KNOB, matches=_never),
)


def coder_failure_classes() -> tuple[FailureClass, ...]:
    """The nine registrable classes, in precedence order — the escape hatch is the mechanism's."""
    return tuple(
        declared.as_failure_class() for declared in CODER_CLASSES if declared.name != OTHER
    )


def register_coder_taxonomy(taxonomy: FailureTaxonomy) -> None:
    """Register this vocabulary against the coder site in ``taxonomy``.

    The mechanism appends its reserved catch-all itself, so the site reports all ten class names.
    """
    taxonomy.register(CODER_SITE_ID, coder_failure_classes())


def classify_coder_failure(error: str) -> str:
    """The name of the first class that recognises ``error``, or :data:`OTHER` when none does.

    Pure and total: any string classifies, no string raises, and an unrecognised one is counted in
    the escape hatch rather than guessed into a class it might not belong to.
    """
    text = error or ""
    for declared in CODER_CLASSES:
        if declared.matches(text):
            return declared.name
    return OTHER


def knob_for(class_name: str) -> str:
    """The knob ``class_name``'s share points at; refuses a name the vocabulary does not declare."""
    for declared in CODER_CLASSES:
        if declared.name == class_name:
            return declared.knob
    declared_names = ", ".join(row.name for row in CODER_CLASSES)
    raise KeyError(f"{class_name!r} is not a coder failure class — declared: {declared_names}")
