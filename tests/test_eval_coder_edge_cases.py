"""The committed edge bucket (#219): fourteen hand-authored briefs at the known-hard extremes.

The canaries (#218) sit at the easy end of every axis so that a red one indicts the harness. This
bucket is their opposite: each case pushes one thing the coder's own feasibility rules name as hard
— deep warmup arithmetic, higher-timeframe base-bar multiplication, a scale-free percentile rule,
the falsify-don't-hope no-trade tape, a revision of an existing file, a reference the library does
not ship — so a benchmark over the corpus measures the coder where it actually fails.

Three properties are held here, and only properties a file can be wrong about:

**Coverage.** Every level of every one of the seven axes appears somewhere in the *committed*
corpus (canary ∪ edge), asserted by enumeration over :data:`AXIS_LEVELS` rather than by a curator's
memory — an axis level no case exercises is a distinction the corpus claims to make and does not.
Each named known-hard shape carries a ``shape:<name>`` tag, and the tag is cross-checked against
the axis level it claims, so a shape label cannot drift away from the difficulty it describes.

**Driver plausibility.** Every brief is the production
:class:`~noctis.research.author.StrategyBrief`, field for field (the loader's own validation), and
every fixed oracle is a :class:`~noctis.strategies.scenario_spec.SpecSuite` the episodic driver's
own FORMULATE parser accepted. For the fixed-oracle cases the brief's ``scenarios`` field is the
faithful rendering of that same spec — exactly what ``_brief_from_formulate`` emits — and the
oracles are run through the **real** spec compiler at a candidate's declared warmup, so the write
gate could stamp them.

**The split stays frozen.** Adding a bucket must not move an already-shipped case's half: the
canary stamps survive a re-stamp of the whole corpus byte for byte, the edge cases carry the
stamps the authoring tool dealt them, and the corpus digest covers both buckets.

The unknown-reference case deserves its own sentence, because it is the one edge case whose honest
outcome is a refusal rather than a file. ``StrategyBrief.reference`` names a library strategy to
adapt, and the author engine rejects one that does not resolve *before* it spends a coder
completion — a bad brief, not a coder failure. The case schema deliberately does not check the
library (it is pure, and a corpus is not a workspace), so a brief may name an absent strategy; this
module pins what that means by asserting the named strategy really is missing from the committed
seed library while the adaptation case's reference really is there.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from noctis.eval.case import Case, ProvenanceKind, Split
from noctis.eval.coder_case import AXIS_LEVELS, REQUIRED_BRIEF_KEYS, Axis, Bucket, coder_payload
from noctis.eval.coder_corpus import (
    CODER_SITE_ID,
    CoderCaseProvider,
    bucket_of,
    load_coder_corpus,
    stamp_splits,
)
from noctis.eval.corpus import HOLDOUT_SHARE, Corpus
from noctis.research.author import StrategyBrief
from noctis.strategies import library
from noctis.strategies.scenario_spec import compile_spec, describe_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_ROOT = REPO_ROOT / "cases"
CODER_ROOT = CASES_ROOT / CODER_SITE_ID
SEED_LIBRARY = REPO_ROOT / "strategies"

#: How many hand-authored briefs the edge bucket ships.
EDGE_CASES = 14

#: How a case declares which known-hard shape it probes, and the axis level such a tag commits to.
SHAPE_TAG_PREFIX = "shape:"
KNOWN_HARD_SHAPES: dict[str, tuple[Axis, str]] = {
    "deep-warmup": (Axis.WARMUP_ARITHMETIC, "composed"),
    "higher-timeframe": (Axis.WARMUP_ARITHMETIC, "higher_timeframe"),
    "scale-free-percentile": (Axis.NO_TRADE_TAPE, "scale_free"),
    "falsified-no-trade": (Axis.NO_TRADE_TAPE, "falsified"),
    "revision": (Axis.COMPOSITION_MODE, "revision"),
    "unknown-reference": (Axis.COMPOSITION_MODE, "reference"),
}

#: A candidate's declared warmup, standing in for the one the write gate resolves from its Params
#: defaults: the fixed oracles are compiled at it here to prove the gate could stamp them.
DECLARED_WARMUP = 60

#: The seed the one honest reference-adaptation case adapts, and the strategy the unknown-reference
#: case names — plausible enough for a driver to have emitted it, absent from every library tier.
ADAPTED_SEED = "donchian_breakout"
ABSENT_REFERENCE = "kama_adaptive_trend"


def _committed() -> tuple[Case, ...]:
    """Every case the repo ships for the coder site, loaded through the real loader."""
    return CoderCaseProvider(cases_root=CASES_ROOT).load(CODER_SITE_ID)


def _edges() -> tuple[Case, ...]:
    """The committed edge bucket."""
    return tuple(case for case in _committed() if bucket_of(case) is Bucket.EDGE)


def _shapes(case: Case) -> tuple[str, ...]:
    """The known-hard shapes one case declares it probes."""
    return tuple(
        tag[len(SHAPE_TAG_PREFIX) :] for tag in case.tags if tag.startswith(SHAPE_TAG_PREFIX)
    )


def _fixed_oracle_cases() -> tuple[Case, ...]:
    """The edge cases that carry a fixed scenario oracle for the write gate to stamp."""
    return tuple(case for case in _edges() if coder_payload(case).scenario_spec is not None)


# ── what the bucket ships ─────────────────────────────────────────────────────────────────


def test_the_edge_bucket_ships_the_fourteen_hand_authored_seed_cases():
    assert len(_edges()) == EDGE_CASES


def test_every_edge_case_loads_through_the_real_provider_and_the_real_coder_validator():
    """A rotted edge case fails the suite here rather than skewing a benchmark number later."""
    edges = _edges()

    assert edges
    assert all(isinstance(coder_payload(case).brief, StrategyBrief) for case in edges)


def test_every_edge_brief_is_field_for_field_valid_strategy_brief_material():
    for case in _edges():
        brief = coder_payload(case).brief

        assert all(getattr(brief, key).strip() for key in REQUIRED_BRIEF_KEYS), case.case_id


def test_every_edge_case_is_labelled_on_all_seven_axes():
    for case in _edges():
        payload = coder_payload(case)

        assert {axis.value for axis in Axis} == set(payload.difficulty), case.case_id


def test_every_edge_case_was_authored_by_hand_rather_than_mined_from_an_operators_run():
    for case in _edges():
        assert case.provenance.kind is ProvenanceKind.AUTHORED, case.case_id
        assert str(case.provenance) == "authored:2026-08-02", case.case_id


# ── coverage: every axis level, every named known-hard shape ──────────────────────────────


def test_every_declared_level_of_every_axis_appears_somewhere_in_the_committed_corpus():
    """The canaries cover the easy ends; the edges owe the rest, and enumeration says so."""
    labelled = [coder_payload(case) for case in _committed()]

    for axis, levels in AXIS_LEVELS.items():
        exercised = {payload.level(axis) for payload in labelled}

        assert set(levels) <= exercised, (axis, sorted(set(levels) - exercised))


def test_every_named_known_hard_shape_is_probed_by_at_least_one_edge_case():
    probed = {shape for case in _edges() for shape in _shapes(case)}

    assert set(KNOWN_HARD_SHAPES) <= probed, sorted(set(KNOWN_HARD_SHAPES) - probed)


def test_every_known_hard_shape_tag_agrees_with_the_axis_level_it_claims():
    """A shape label that outran its own difficulty label would make the corpus unreadable."""
    for case in _edges():
        payload = coder_payload(case)

        for shape in _shapes(case):
            assert shape in KNOWN_HARD_SHAPES, (case.case_id, shape)
            axis, level = KNOWN_HARD_SHAPES[shape]
            assert payload.level(axis) == level, (case.case_id, shape)


def test_the_edge_bucket_reaches_the_hardest_level_of_every_axis_that_declares_one():
    """The bucket's whole point: no canary sits at a hardest level, so an edge case must."""
    levels = {axis: {coder_payload(case).level(axis) for case in _edges()} for axis in Axis}

    for axis, exercised in levels.items():
        assert AXIS_LEVELS[axis][-1] in exercised, axis


# ── the oracle mix ────────────────────────────────────────────────────────────────────────


def test_the_edge_bucket_mixes_both_oracle_modes_with_at_least_four_fixed_oracles():
    modes = [coder_payload(case).level(Axis.ORACLE_MODE) for case in _edges()]

    assert modes.count("fixed_spec") >= 4
    assert modes.count("authored") >= 4


def test_a_fixed_oracle_case_carries_a_spec_the_drivers_own_formulate_parser_accepted():
    cases = _fixed_oracle_cases()

    assert len(cases) >= 4
    assert all(coder_payload(case).scenario_spec.scenarios for case in cases)


def test_every_fixed_oracle_brief_states_its_scenarios_as_the_rendering_of_that_same_spec():
    """The driver renders the fixed oracle into the brief; a case written any other way is
    measuring a translation layer instead of the coder."""
    for case in _fixed_oracle_cases():
        payload = coder_payload(case)

        assert payload.brief.scenarios == describe_spec(payload.scenario_spec), case.case_id


def test_every_fixed_oracle_compiles_through_the_real_spec_compiler_at_a_declared_warmup():
    """What the write gate would do with the spec, done here: an oracle that cannot compile is
    an edge case the gate would reject before the coder ever saw it."""
    for case in _fixed_oracle_cases():
        suite = coder_payload(case).scenario_spec

        compiled = compile_spec(suite, DECLARED_WARMUP)

        assert [scenario.name for scenario in compiled] == [
            spec.name for spec in suite.scenarios
        ], case.case_id


def test_no_edge_case_labelled_authored_smuggles_a_fixed_oracle_past_the_write_gate():
    for case in _edges():
        payload = coder_payload(case)

        if payload.level(Axis.ORACLE_MODE) == "authored":
            assert payload.scenario_spec is None, case.case_id


# ── the two reference cases: one that resolves, one that honestly does not ────────────────


def test_the_reference_adaptation_case_names_a_strategy_the_committed_seed_library_ships():
    referenced = {
        coder_payload(case).brief.reference
        for case in _edges()
        if coder_payload(case).level(Axis.COMPOSITION_MODE) == "reference"
    }

    assert ADAPTED_SEED in referenced
    assert library.strategy_path(SEED_LIBRARY, ADAPTED_SEED) is not None


def test_the_unknown_reference_case_names_a_strategy_no_library_tier_can_resolve():
    """An unknown reference is a bad brief the author engine refuses before spending a
    completion, so the case only means what it says while the name really is absent."""
    (case,) = tuple(case for case in _edges() if "unknown-reference" in _shapes(case))

    assert coder_payload(case).brief.reference == ABSENT_REFERENCE
    assert library.strategy_path(SEED_LIBRARY, ABSENT_REFERENCE) is None


# ── the split: dealt once, and the canaries never moved ───────────────────────────────────


def test_every_edge_case_carries_the_split_the_authoring_tool_stamped_into_its_file():
    for case in _edges():
        assert case.split in (Split.TUNING, Split.HOLDOUT), case.case_id


def test_the_edge_bucket_sits_within_one_case_of_the_holdout_share():
    edges = _edges()
    holdout = sum(1 for case in edges if case.split is Split.HOLDOUT)

    assert abs(holdout - len(edges) * HOLDOUT_SHARE) <= 1


def test_re_stamping_the_committed_corpus_writes_nothing_because_every_case_is_frozen():
    assert stamp_splits(CASES_ROOT) == ()


def test_stamping_the_new_edge_cases_leaves_every_canary_file_byte_for_byte_alone(tmp_path):
    """The property the stamp buys, exercised over the real corpus: growing it re-deals nothing."""
    shutil.copytree(CASES_ROOT, tmp_path / "cases")
    grown = tmp_path / "cases" / CODER_SITE_ID
    canaries = {path: path.read_bytes() for path in sorted((grown / "canary").glob("*.yaml"))}
    for path in sorted((grown / "edge").glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        document.pop("split")
        path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")

    stamped = stamp_splits(tmp_path / "cases")

    assert len(stamped) == EDGE_CASES
    assert {path: path.read_bytes() for path in canaries} == canaries


def test_re_stamping_a_stripped_edge_bucket_deals_every_case_the_half_its_file_declares(tmp_path):
    shutil.copytree(CASES_ROOT, tmp_path / "cases")
    grown = tmp_path / "cases" / CODER_SITE_ID
    for path in sorted((grown / "edge").glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        document.pop("split")
        path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    stamp_splits(tmp_path / "cases")

    redealt = CoderCaseProvider(cases_root=tmp_path / "cases").load(CODER_SITE_ID)

    assert {case.case_id: case.split for case in redealt} == {
        case.case_id: case.split for case in _committed()
    }


def test_the_committed_corpus_digest_covers_the_edge_bucket_and_not_the_canaries_alone():
    canaries = tuple(case for case in _committed() if bucket_of(case) is Bucket.CANARY)

    whole = load_coder_corpus(CASES_ROOT)

    assert whole.identity.case_count == len(canaries) + EDGE_CASES
    assert whole.digest != Corpus(CODER_SITE_ID, canaries).digest
