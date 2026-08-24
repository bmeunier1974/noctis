"""The coder case schema — a benchmark job shaped exactly like a real authoring job.

The coder site is asked for one thing: turn a research brief into one complete strategy file that
survives the fresh-subprocess write gate. A benchmark case for it must therefore *be* a brief —
not a paraphrase of one, not a prompt fragment — because the moment a case carries a field
production does not, or misses one production requires, the benchmark stops measuring the coder and
starts measuring a translation layer nobody reviews. So the payload's fields are
:class:`~noctis.research.author.StrategyBrief`'s own (:data:`BRIEF_KEYS`, asserted field-for-field
against the production dataclass in ``tests/test_eval_coder_cases.py``, so drift on either side
breaks the build), and the optional fixed oracle is read by the strategy layer's own model-dialect
parse, so a case's ``scenario_spec`` is one a real FORMULATE could have emitted or it is refused
here.

**No expected output, ever — and the coder's own spellings of that mistake are named.** The eval
core refuses the generic shapes (``expected``, ``solution``, ``gold``, …) at the top level of a case
document; this schema extends the refusal *into the payload*, where a curator would actually be
tempted to park one, and adds the coder-specific spellings (:data:`CODER_REFUSED_KEYS`:
``expected_source``, ``reference_solution``, ``model_answer``, …). The write gate is the
expectation. A stored file would be "correct" only until the strategy contract improved.

**The bucket is a tag, because a bucket is what a case is *for*, not how hard it is.** A case
declares exactly one ``bucket:<name>`` tag from a closed vocabulary (:class:`Bucket`):

* ``field`` — harvested from a real run: production actually asked the coder this;
* ``replay`` — a curated re-run of a brief with a known gate rejection, kept so a fix stays fixed;
* ``edge`` — a deliberately hard brief that pushes one difficulty axis to its extreme;
* ``canary`` — a brief so plain that a failure indicts the harness rather than the model.

**Seven difficulty axes, each a closed value set, and every case is labelled on all seven.** The
corpus stratifies its tuning/holdout split on the difficulty mapping, so a partly-labelled case
would land in a stratum of its own and quietly skew the split; and an axis whose values are open
text is an axis two curators spell differently. The vocabulary (:class:`Axis`,
:data:`AXIS_LEVELS`) — every level below is a distinction the *production* authoring path already
makes, never a scale invented to look thorough:

* ``composition_mode``: ``scratch`` | ``reference`` | ``revision`` — the three ways the author
  engine composes a prompt (author from nothing, adapt a named library strategy, revise the file
  that already carries the target name).
* ``oracle_mode``: ``authored`` | ``fixed_spec`` — the write gate is tolerant-both: the coder
  writes its own ``scenarios()``, or the gate machine-stamps a fixed :class:`SpecSuite` and rejects
  any ``scenarios()`` the coder writes.
* ``warmup_arithmetic``: ``single`` | ``composed`` | ``higher_timeframe`` — how hard it is to derive
  warmup from the ``Params`` defaults: one lookback, several composed into the longest, or a
  higher-timeframe strategy whose warmup multiplies by the bars per decision bar.
* ``state_complexity``: ``stateless`` | ``rolling`` | ``latched`` — nothing kept between bars, an
  incremental rolling window/accumulator, or rolling state plus a position latch.
* ``no_trade_tape``: ``trivial`` | ``falsified`` | ``scale_free`` — how hard the no-trade tape is to
  satisfy: a tape that cannot trigger, a tape that must falsify the level condition, or a scale-free
  percentile-rank rule no amplitude can silence (the feasibility rules' hardest case).
* ``param_space_breadth``: ``narrow`` | ``moderate`` | ``broad`` — one or two tuned parameters,
  three or four, or five and up.
* ``api_surface``: ``bars_only`` | ``indicators`` | ``exits`` — how much of the API contract sheet
  the file must wire: bars and targets alone, plus the indicator surface (tail functions / State
  classes), plus protective ``ExitRules`` on top.

**A label that disagrees with its payload is refused**, so the axes stay measurements rather than
decoration: a ``reference`` composition with no ``reference`` in the brief, a ``fixed_spec`` oracle
with no ``scenario_spec``, an ``authored`` oracle that ships one anyway (the gate would stamp it and
reject the coder's own tapes — the label would lie about which path was measured).

**Strict, loud, one-pass** — the eval core's posture, extended: every defect in a file, generic and
coder-specific, comes back in one :class:`~noctis.eval.case.MalformedCase` naming the file, so a
broken case is one edit rather than a fix-one-rerun loop. And **pure**: validation over data, no
directory, no clock. :class:`~noctis.eval.case_provider.YamlCaseProvider` stays the one thing in the
layer that reads, and :func:`coder_payload` is what its output is projected through.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, NamedTuple

from noctis.eval.case import (
    REFUSED_KEYS,
    Case,
    MalformedCase,
    Provenance,
    Split,
    parse_case,
)
from noctis.research.author import StrategyBrief
from noctis.strategies.scenario_spec import (
    PARSE_WARM,
    SpecError,
    SpecSuite,
    compile_spec,
    spec_from_payload,
)

__all__ = [
    "AXIS_LEVELS",
    "BRIEF_KEYS",
    "CODER_AXES",
    "BUCKET_TAG_PREFIX",
    "CODER_REFUSED_KEYS",
    "PAYLOAD_KEYS",
    "REQUIRED_BRIEF_KEYS",
    "SPEC_KEY",
    "Axis",
    "Bucket",
    "CoderPayload",
    "coder_payload",
    "parse_coder_case",
]

# The brief the coder consumes, field for field. Declared here rather than derived from the
# dataclass on purpose: a *declaration* can be compared against production, and the comparison is
# the test that breaks the build when either side moves.
BRIEF_KEYS: tuple[str, ...] = (
    "thesis",
    "entry_exit",
    "param_space",
    "scenarios",
    "reference",
    "style",
    "symbols",
)

# The four the author engine has no default for — the division-of-labor guard restated: a case
# missing one of these would ask the coder to supply research judgment the brief owes it.
REQUIRED_BRIEF_KEYS: tuple[str, ...] = ("thesis", "entry_exit", "param_space", "scenarios")

# The optional fixed scenario oracle, in the FORMULATE emit's own vocabulary.
SPEC_KEY = "scenario_spec"

# Everything a coder payload may declare. Anything else is refused by name.
PAYLOAD_KEYS: tuple[str, ...] = (*BRIEF_KEYS, SPEC_KEY)


class Bucket(StrEnum):
    """What a coder case is *for* — the closed vocabulary a corpus is composed from."""

    FIELD = "field"
    REPLAY = "replay"
    EDGE = "edge"
    CANARY = "canary"


# How a case declares its bucket: one tag, so the bucket rides the case's own label slot rather
# than pretending to be an eighth difficulty axis or a field of the site's input.
BUCKET_TAG_PREFIX = "bucket:"


class Axis(StrEnum):
    """The seven difficulty axes a coder case is labelled on. Every case declares all seven."""

    COMPOSITION_MODE = "composition_mode"
    ORACLE_MODE = "oracle_mode"
    WARMUP_ARITHMETIC = "warmup_arithmetic"
    STATE_COMPLEXITY = "state_complexity"
    NO_TRADE_TAPE = "no_trade_tape"
    PARAM_SPACE_BREADTH = "param_space_breadth"
    API_SURFACE = "api_surface"


#: Each axis's closed value set, ordered easiest first. Every level names a distinction the
#: production authoring path already makes (see the module docstring for what each one means).
AXIS_LEVELS: Mapping[Axis, tuple[str, ...]] = MappingProxyType(
    {
        Axis.COMPOSITION_MODE: ("scratch", "reference", "revision"),
        Axis.ORACLE_MODE: ("authored", "fixed_spec"),
        Axis.WARMUP_ARITHMETIC: ("single", "composed", "higher_timeframe"),
        Axis.STATE_COMPLEXITY: ("stateless", "rolling", "latched"),
        Axis.NO_TRADE_TAPE: ("trivial", "falsified", "scale_free"),
        Axis.PARAM_SPACE_BREADTH: ("narrow", "moderate", "broad"),
        Axis.API_SURFACE: ("bars_only", "indicators", "exits"),
    }
)

#: The seven axes flat, in declaration order — the same vocabulary as :class:`Axis`, as the plain
#: strings a record's keys and a declaration's ``difficulty_axes`` are made of. Spelled once here,
#: where the enum lives, so the site declaration and the site's own reading share one object (#306).
CODER_AXES: tuple[str, ...] = tuple(axis.value for axis in Axis)

#: The expected-output shapes a coder case is refused for, at the top level **and** inside the
#: payload: the eval core's generic set plus the spellings this site invites. ``reference`` is a
#: real brief field, so only its answer-shaped compounds are named.
CODER_REFUSED_KEYS: frozenset[str] = REFUSED_KEYS | {
    "canonical_solution",
    "expected_code",
    "expected_file",
    "expected_source",
    "expected_strategy",
    "golden_source",
    "model_answer",
    "reference_answer",
    "reference_code",
    "reference_implementation",
    "reference_solution",
    "reference_source",
    "reference_strategy",
    "solution_source",
}

_NO_EXPECTATION = (
    "a coder case carries no expected output; the fresh-subprocess write gate is this site's "
    "oracle, so a corpus file never rots into disagreement with it"
)

# The spellings this site adds to the eval core's own refused set. Only these are named here: the
# generic shapes are already refused, by name and with the same reason, one layer down.
_CODER_ONLY_REFUSED: frozenset[str] = CODER_REFUSED_KEYS - REFUSED_KEYS

# Only reachable if a defect went unreported — the same defensive spelling the eval core uses.
_UNREADABLE = "the coder half of this case could not be read"


@dataclass(frozen=True)
class CoderPayload:
    """One coder case, typed: the brief the author engine consumes, and how it is labelled.

    ``brief`` is the production :class:`~noctis.research.author.StrategyBrief` itself, so a harness
    hands a benchmark job to the same engine a session does; ``scenario_spec`` is the fixed oracle
    when the case declares one; ``bucket``, ``difficulty`` and ``tags`` are what a corpus
    stratifies and reports over; ``provenance`` and ``split`` ride straight from the eval core's
    case model, which owns both.
    """

    case_id: str
    brief: StrategyBrief
    bucket: Bucket
    difficulty: Mapping[str, str]
    provenance: Provenance
    scenario_spec: SpecSuite | None = None
    split: Split | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Hold the labels by value and read-only: a payload is a record of one ask."""
        object.__setattr__(self, "difficulty", MappingProxyType(dict(self.difficulty)))
        object.__setattr__(self, "tags", tuple(self.tags))

    def level(self, axis: Axis) -> str:
        """This case's level on one difficulty axis — always present, always in its value set."""
        return self.difficulty[axis]


def coder_payload(case: Case, *, source: str | None = None) -> CoderPayload:
    """The typed coder reading of an already-loaded case, or a refusal naming every defect.

    This is the provider path: :class:`~noctis.eval.case_provider.YamlCaseProvider` validates the
    generic half of a file and hands back a :class:`~noctis.eval.case.Case`; this projects that case
    onto the coder schema. ``source`` is what a refusal names — pass the file path the case came
    from; without one the refusal names the case id, which the YAML layout takes from the file name.
    """
    label = source or f"case {case.case_id!r}"
    reading, defects = _read(case.payload, case.difficulty, case.tags)
    if reading is None or defects:
        raise MalformedCase(f"{label}: {'; '.join(defects or (_UNREADABLE,))}")
    return _payload(case.case_id, case.provenance, case.split, reading)


def parse_coder_case(
    document: object,
    *,
    case_id: str,
    source: str,
    declared_site_ids: Collection[str] | None = None,
) -> CoderPayload:
    """Read one coder case document end to end — generic half and coder half — in a single pass.

    Both halves' defects come back in one :class:`~noctis.eval.case.MalformedCase` naming
    ``source``, because a curator fixing a file wants to read every problem with it once. The
    generic validation is :func:`~noctis.eval.case.parse_case`'s, unchanged and never re-implemented
    here; ``declared_site_ids`` is forwarded to it verbatim.

    The one thing this adds at the *document* level is the coder's own expected-output spellings
    (:data:`CODER_REFUSED_KEYS` beyond the core's): the core would call ``expected_source`` merely
    an unknown key, and a curator reading that would rename it rather than delete it.
    """
    generic: list[str] = []
    case: Case | None = None
    try:
        case = parse_case(
            document, case_id=case_id, source=source, declared_site_ids=declared_site_ids
        )
    except MalformedCase as refusal:
        generic.append(_undecorated(str(refusal), source))

    reading: _Reading | None = None
    coder: list[str] = []
    if isinstance(document, Mapping):
        smuggled = (str(key) for key in document if str(key).lower() in _CODER_ONLY_REFUSED)
        coder = [f"key {key!r} is refused — {_NO_EXPECTATION}" for key in sorted(smuggled)]
        reading, payload_defects = _read(
            document.get("payload"), document.get("difficulty", {}), document.get("tags", ())
        )
        coder.extend(payload_defects)
    defects = [*generic, *coder]
    if case is None or reading is None or defects:
        raise MalformedCase(f"{source}: {'; '.join(defects or (_UNREADABLE,))}")
    return _payload(case.case_id, case.provenance, case.split, reading)


class _Reading(NamedTuple):
    """The coder half of a case, once every part of it has been read without a defect."""

    brief: StrategyBrief
    scenario_spec: SpecSuite | None
    bucket: Bucket
    difficulty: Mapping[str, str]
    tags: tuple[str, ...]


def _payload(
    case_id: str, provenance: Provenance, split: Split | None, reading: _Reading
) -> CoderPayload:
    """One validated reading, plus the eval core's own fields, as the typed payload."""
    return CoderPayload(
        case_id=case_id,
        brief=reading.brief,
        bucket=reading.bucket,
        difficulty=reading.difficulty,
        provenance=provenance,
        scenario_spec=reading.scenario_spec,
        split=split,
        tags=reading.tags,
    )


def _read(payload: object, difficulty: object, tags: object) -> tuple[_Reading | None, list[str]]:
    """The whole coder-specific validation, in one pass over the three parts that carry it.

    A part whose *generic* shape is already wrong (a payload that is not a mapping, tags that are
    not strings) is skipped rather than re-refused: only :func:`parse_coder_case` can reach this
    with one, and its generic half has already named it. A :class:`~noctis.eval.case.Case` always
    carries all three well-shaped, so nothing is skipped on the provider path.
    """
    brief: StrategyBrief | None = None
    spec: SpecSuite | None = None
    brief_defects: list[str] = []
    if isinstance(payload, Mapping):
        brief, spec, brief_defects = _brief(payload)
    bucket, bucket_defects, rest = _bucket(tags)
    axes, axis_defects = _axes(difficulty)
    defects = [*brief_defects, *bucket_defects, *axis_defects]
    if brief is not None and axes is not None and isinstance(payload, Mapping):
        defects.extend(_incoherent(brief, SPEC_KEY in payload, axes))
    if defects or brief is None or bucket is None or axes is None:
        return None, defects
    return _Reading(brief, spec, bucket, axes, rest), defects


def _brief(payload: Mapping[Any, Any]) -> tuple[StrategyBrief | None, SpecSuite | None, list[str]]:
    """A payload as the production brief plus its optional fixed oracle, or the defects refusing it.

    The brief is built with the author engine's own type, so a field production stops requiring
    stops being required here the moment the mirror test is re-run.
    """
    defects = _payload_key_defects(payload)
    values: dict[str, Any] = {}
    for key in REQUIRED_BRIEF_KEYS:
        if key not in payload:
            continue
        text = payload[key]
        if not isinstance(text, str) or not text.strip():
            defects.append(f"brief field {key!r} must be a non-empty string, got {text!r}")
        else:
            values[key] = text
    for key in ("reference", "style"):
        text = payload.get(key)
        if text is None:
            continue
        if not isinstance(text, str) or not text.strip():
            defects.append(f"brief field {key!r} must be a non-empty string, got {text!r}")
        else:
            values[key] = text
    symbols, symbol_defect = _symbols(payload.get("symbols"))
    if symbol_defect:
        defects.append(symbol_defect)
    else:
        values["symbols"] = symbols
    spec, spec_defect = _spec(payload)
    if spec_defect:
        defects.append(spec_defect)
    if defects:
        return None, None, defects
    return StrategyBrief(**values), spec, defects


def _payload_key_defects(payload: Mapping[Any, Any]) -> list[str]:
    """The refused, unknown and missing keys of a coder payload, as messages naming each one."""
    refused = sorted(str(key) for key in payload if str(key).lower() in CODER_REFUSED_KEYS)
    defects = [f"payload key {key!r} is refused — {_NO_EXPECTATION}" for key in refused]
    unknown = sorted(
        str(key)
        for key in payload
        if str(key) not in PAYLOAD_KEYS and str(key).lower() not in CODER_REFUSED_KEYS
    )
    if unknown:
        defects.append(
            f"unknown payload key(s) {', '.join(repr(key) for key in unknown)} — a coder payload "
            f"declares {', '.join(PAYLOAD_KEYS)}"
        )
    missing = [key for key in REQUIRED_BRIEF_KEYS if key not in payload]
    if missing:
        defects.append(
            f"missing brief field(s) {', '.join(repr(key) for key in missing)} — the brief is the "
            "coder's whole research input, and the coder never invents one"
        )
    return defects


def _symbols(candidate: object) -> tuple[tuple[str, ...], str | None]:
    """A payload's researched symbols as a tuple, or the defect that stops them being one."""
    if candidate is None:
        return (), None
    if isinstance(candidate, str) or not isinstance(candidate, Sequence):
        return (), f"brief field 'symbols' must be a list of ticker strings, got {candidate!r}"
    if not all(isinstance(symbol, str) and symbol.strip() for symbol in candidate):
        return (), f"brief field 'symbols' must be a list of ticker strings, got {candidate!r}"
    return tuple(candidate), None


def _spec(payload: Mapping[Any, Any]) -> tuple[SpecSuite | None, str | None]:
    """The declared fixed oracle as a :class:`SpecSuite`, or the defect refusing its shape.

    Read by :func:`~noctis.strategies.scenario_spec.spec_from_payload` and compiled at
    :data:`~noctis.strategies.scenario_spec.PARSE_WARM` — the strategy layer's own model-dialect
    parse and the same parse-time structural compile a real FORMULATE emit passes — so a case's
    oracle is one production could have produced. Straight to the layer that owns what a spec *is*,
    never through the LLM driver, whose only business in an emit is which field carries the spec
    (#319): a bench case is data already, so it needs the parse, not the transport. Full
    compilation at a candidate's real declared warmup stays the write gate's job.
    """
    if SPEC_KEY not in payload:
        return None, None
    try:
        suite = spec_from_payload(_plain(payload[SPEC_KEY]))
        compile_spec(suite, PARSE_WARM)
    except (SpecError, ValueError, TypeError, KeyError) as refusal:
        return None, f"{SPEC_KEY} is malformed — {refusal}"
    return suite, None


def _bucket(tags: object) -> tuple[Bucket | None, list[str], tuple[str, ...]]:
    """A case's bucket, the defects refusing it, and the tags left over once it is consumed."""
    if isinstance(tags, str) or not isinstance(tags, Sequence):
        return None, [], ()
    labels = tuple(str(tag) for tag in tags)
    declared = [tag for tag in labels if tag.startswith(BUCKET_TAG_PREFIX)]
    vocabulary = ", ".join(bucket.value for bucket in Bucket)
    rest = tuple(tag for tag in labels if tag not in declared)
    if len(declared) != 1:
        return (
            None,
            [
                f"declare exactly one {BUCKET_TAG_PREFIX}<name> tag — one of {vocabulary} — got "
                f"{', '.join(repr(tag) for tag in declared) or 'none'}"
            ],
            rest,
        )
    name = declared[0][len(BUCKET_TAG_PREFIX) :]
    try:
        return Bucket(name), [], rest
    except ValueError:
        return None, [f"unknown bucket {name!r} — a coder case is one of {vocabulary}"], rest


def _axes(difficulty: object) -> tuple[Mapping[str, str] | None, list[str]]:
    """A case's seven difficulty levels, or the defects naming every axis it got wrong."""
    if not isinstance(difficulty, Mapping) or not all(
        isinstance(axis, str) and isinstance(level, str) for axis, level in difficulty.items()
    ):
        return None, []
    declared = ", ".join(axis.value for axis in Axis)
    defects: list[str] = []
    unknown = sorted(key for key in difficulty if key not in set(Axis))
    if unknown:
        defects.append(
            f"unknown difficulty axis/axes {', '.join(repr(key) for key in unknown)} — a coder "
            f"case is labelled on {declared}"
        )
    missing = [axis.value for axis in Axis if axis.value not in difficulty]
    if missing:
        defects.append(
            f"missing difficulty axis/axes {', '.join(repr(key) for key in missing)} — a coder "
            f"case is labelled on all seven: {declared}"
        )
    for axis in Axis:
        level = difficulty.get(axis.value)
        if level is not None and level not in AXIS_LEVELS[axis]:
            defects.append(
                f"difficulty axis {axis.value!r} value {level!r} is not one of "
                f"{', '.join(AXIS_LEVELS[axis])}"
            )
    if defects:
        return None, defects
    return dict(difficulty), defects


def _incoherent(brief: StrategyBrief, declares_spec: bool, axes: Mapping[str, str]) -> list[str]:
    """The defects of a case whose difficulty labels disagree with the brief they describe.

    An axis is a measurement, so it may not merely *claim* a composition or an oracle the payload
    cannot express: a benchmark that stratified on those labels would report the wrong thing.
    """
    defects: list[str] = []
    if axes[Axis.COMPOSITION_MODE] == "reference" and not brief.reference:
        defects.append(
            "composition_mode 'reference' but the brief names no 'reference' strategy to adapt"
        )
    if axes[Axis.ORACLE_MODE] == "fixed_spec" and not declares_spec:
        defects.append(f"oracle_mode 'fixed_spec' but the payload declares no {SPEC_KEY!r}")
    if axes[Axis.ORACLE_MODE] == "authored" and declares_spec:
        defects.append(
            f"oracle_mode 'authored' but the payload declares a {SPEC_KEY!r} — the write gate "
            "would stamp it and reject any scenarios() the coder authored"
        )
    return defects


def _undecorated(refusal: str, source: str) -> str:
    """One refusal's defects without the source label it already carries, so it composes."""
    prefix = f"{source}: "
    return refusal[len(prefix) :] if refusal.startswith(prefix) else refusal


def _plain(value: Any) -> Any:
    """A deep-frozen payload value back as plain data — read-only maps and tuples undone.

    A loaded case's payload is frozen all the way down, and the model-dialect parse reads the
    literal shapes a transport hands it (``dict``, ``list``), so the oracle is thawed on its way
    into the production parser rather than the parser being taught a second set of shapes.
    """
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value
