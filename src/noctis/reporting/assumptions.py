"""The honesty block — the arena a run's numbers were produced in (story #143, epic #126).

Slippage and commission drag can halve a backtested return; a walk-forward geometry, a holdout
size or a promotion threshold moved by one notch produces a different funnel entirely. A results
page that does not state its assumptions is not taken seriously — and a page that states them in
prose cannot be diffed. So the record carries them as **data**: one block a website renders as a
table, and two runs' blocks subtract.

Three rules hold this honest, and they are why this module exists at all rather than a dict
literal inside :func:`~noctis.reporting.run_record.build`:

* **Derived, never transcribed.** Everything the run *configured* — the fill costs, both holdout
  sizes, the exhaustion gate's ``min_trials``, every promotion threshold — is read out of the
  run's own frozen ``inputs`` (:func:`noctis.config.rehydrate.freeze_inputs`), which is the same
  settings dump the engine rehydrates and obeys. A knob added to ``promotion`` tomorrow appears
  here with no edit, because the whole subtree is published rather than a hand-kept list of it.
* **What must be named is held to what it names.** Three things are engine constants rather than
  settings — the fill model, the walk-forward sizing bounds, and the benchmark's rebalancing
  convention. They are spelled once here, and ``tests/test_run_assumptions.py`` holds each to the
  code that implements it: a real fill lands at bar t+1's open, ``PipelineConfig.auto`` sizes
  every split inside the stated bounds (at both ends), and the equal-weight basket really does
  drift instead of rebalancing.
* **The gate is measured, not claimed.** ``paper_only`` and ``real_orders_reachable`` are read off
  the gate's own resolved verdict, frozen as ``inputs.execution_mode`` (story #132) — the record
  never carries ``mode`` or ``allow_live`` themselves, because that pair must have exactly two
  independent sources and a record is neither of them (AGENTS.md rule 1). A run that froze no
  verdict says ``null``: "nobody measured" is a different fact from "paper", and only one of them
  is a claim this block is entitled to make.

Pure, and stdlib-only apart from the benchmark constants next door: it takes the frozen inputs as
a mapping and returns a document. No config import (the run store must stay importable on the
core install), no clock, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping

from noctis.reporting.metrics import BENCHMARK_METHOD, BENCHMARK_NAME

__all__ = [
    "FILL_MODEL",
    "LOOKAHEAD",
    "MAX_TEST_BARS",
    "MAX_TRAIN_BARS",
    "MIN_TEST_BARS",
    "MIN_TRAIN_BARS",
    "PAPER_MODE",
    "REBALANCING",
    "assumptions",
]

# How every simulated trade is priced, in one word: the decision taken on bar *t* is executed at
# bar *t+1*'s **open** (``broker/simulator.py``). Named rather than derived — there is no setting
# behind it — and pinned by a behavioural test that watches a real fill land on the next open.
FILL_MODEL = "next_bar_open"
LOOKAHEAD = (
    "none — a decision taken on bar t is filled at bar t+1's open, and every walk-forward test "
    "window sits strictly after its train window"
)

# The fill costs are charged **per side**, so a round trip pays both legs. Stated because the
# single most common way to misread a cost assumption is to halve it.
COST_BASIS = "per side — a round trip pays both legs"

# The walk-forward sizing bounds ``PipelineConfig.auto`` works within. The geometry itself is
# per-candidate (it is sized from the panel's shortest series), so a run-level record cannot state
# one number honestly — it states the rule, and a test holds the rule to the heuristic, at both
# ends, so moving a bound is a red test rather than silent drift.
MIN_TRAIN_BARS = 40
MAX_TRAIN_BARS = 120
MIN_TEST_BARS = 20
MAX_TEST_BARS = 40

# The benchmark's rebalancing convention (story #142's ``_equal_weight_levels``), stated so the
# comparison is reproducible and nobody mistakes it for an index.
REBALANCING = (
    "none — weights are set at the first session mark this run can price and are never "
    "rebalanced; each name's contribution drifts with its own return thereafter"
)

# The gate verdict that means real-money orders are unreachable. The record carries the verdict,
# never the two settings behind it.
PAPER_MODE = "paper"

# Where the run's champions, paper account and journals live: inside the run's own tree, so two
# runs never share a board and a comparison between them is a comparison of two experiments.
STATE_SCOPE = "run"

# The keys inside the frozen settings dump this block reads. Spelled once, so the derivation is
# reviewable in one place.
_BACKTEST = "backtest"
_RESEARCH = "research"
_PROMOTION = "promotion"
_CHAMPION_COUNT = "champion_count"


def assumptions(inputs: Mapping[str, object] | None) -> dict:
    """The arena this run ran in, as one machine-readable block.

    ``inputs`` is the record's frozen configuration (``None`` for an adopted history that never
    froze one). Facts the *engine* fixes are stated for every run; facts the *run* configured are
    stated when the run froze them and are explicit ``null`` when it did not — an omitted key
    would leave a reader unable to tell "not applicable" from "this schema version lacked it".
    """
    resolved = _resolved(inputs)
    backtest = _section(resolved, _BACKTEST)
    research = _section(resolved, _RESEARCH)
    fee_bps = _number(backtest.get("fee_bps"))
    slippage_bps = _number(backtest.get("slippage_bps"))
    return {
        # ── the live-money double gate, as the gate itself resolved it ──────────────────────
        "paper_only": _paper_only(inputs),
        "live_gate": _live_gate(inputs),
        # ── how a simulated trade is priced ─────────────────────────────────────────────────
        "fill_model": FILL_MODEL,
        "lookahead": LOOKAHEAD,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "round_trip_cost_bps": _round_trip_bps(fee_bps, slippage_bps),
        "costs_charged": COST_BASIS,
        # ── how a candidate is judged out of sample ─────────────────────────────────────────
        "walk_forward": {
            # Per candidate, from the panel's shortest series — so the record states the rule
            # and its bounds rather than one number that would be a lie for every other panel.
            "sizing": "auto",
            "min_train_bars": MIN_TRAIN_BARS,
            "max_train_bars": MAX_TRAIN_BARS,
            "min_test_bars": MIN_TEST_BARS,
            "max_test_bars": MAX_TEST_BARS,
            # ``null`` = one test window: consecutive splits advance by exactly the test size, so
            # test windows never overlap.
            "step_bars": None,
            "embargo_bars": 0,
            "test_after_train": True,
        },
        "forward_holdout": {
            "min_bars": MIN_TEST_BARS,
            "max_bars": MAX_TEST_BARS,
            "reserved": True,
            "note": "one test window of the most-recent bars, reserved from the pre-filter, the "
            "walk-forward and the optimizer alike; a panel too short to leave a full split "
            "behind degrades to no forward-holdout gate rather than starving the search",
        },
        "symbol_holdout": {
            "size": _int(research.get("symbol_holdout_size")),
            "fit_set_size": _int(research.get("fit_set_size")),
            # Which names were held out is sampled per research session, not fixed for the run.
            "symbols": None,
        },
        "min_trials": _int(research.get("min_trials")),
        "promotion_thresholds": _promotion_thresholds(resolved),
        # ── what the realised numbers are compared against ──────────────────────────────────
        "benchmark": {
            "name": BENCHMARK_NAME,
            "method": BENCHMARK_METHOD,
            "rebalancing": REBALANCING,
        },
        "state_scope": STATE_SCOPE,
    }


def _paper_only(inputs: Mapping[str, object] | None) -> bool | None:
    """Whether real-money orders were unreachable — **measured**, never asserted.

    The measurement is the safety gate's own verdict for this run, frozen at creation and refused
    a second source by construction (:mod:`noctis.config.rehydrate`): ``mode`` and ``allow_live``
    are never written to a record and never restored from one, so this block reads the outcome the
    gate reached rather than re-deciding it. ``None`` for a run that froze no verdict — an adopted
    history — because "nobody measured" is not "paper".
    """
    mode = _execution_mode(inputs)
    return None if mode is None else mode == PAPER_MODE


def _live_gate(inputs: Mapping[str, object] | None) -> dict:
    """The gate's verdict, and the sentence that says why the record holds only the verdict."""
    mode = _execution_mode(inputs)
    return {
        "execution_mode": mode,
        "real_orders_reachable": None if mode is None else mode != PAPER_MODE,
        # The gate re-resolves from config.yaml + ALLOW_LIVE at every process start, so a resumed
        # run is gated by the machine it is resumed on, never by what this record says.
        "re_resolved_each_segment": True,
        "note": "real-money orders need BOTH config mode: live and ALLOW_LIVE=true; the pair is "
        "re-resolved from the config file and the environment at every process start, is never "
        "written to this record and is never restored from one, and a resume whose verdict "
        "differs from the frozen one is refused rather than reconciled",
    }


def _execution_mode(inputs: Mapping[str, object] | None) -> str | None:
    if not isinstance(inputs, Mapping):
        return None
    mode = inputs.get("execution_mode")
    return mode if isinstance(mode, str) and mode else None


def _promotion_thresholds(resolved: Mapping[str, object]) -> dict | None:
    """Every promotion knob this run froze, verbatim, plus the board size they are applied to.

    The **whole** ``promotion`` subtree, not a curated list of it: ``PromotionRules.from_settings``
    is the one mapping from these settings to the gates, and publishing the subtree means a
    threshold added there lands here with no edit and can never quietly go unpublished. The board
    size lives one level up in the settings and travels with them, because "beat the weakest of
    three" and "beat the weakest of ten" are different gates.
    """
    if not resolved:
        return None
    promotion = _section(resolved, _PROMOTION)
    if not promotion:
        return None
    thresholds: dict[str, object] = dict(promotion)
    thresholds[_CHAMPION_COUNT] = _int(resolved.get(_CHAMPION_COUNT))
    return thresholds


def _round_trip_bps(fee_bps: float | None, slippage_bps: float | None) -> float | None:
    """What a full round trip costs: both legs, both charges. ``None`` if either is unknown."""
    if fee_bps is None or slippage_bps is None:
        return None
    return round(2.0 * (fee_bps + slippage_bps), 10)


def _resolved(inputs: Mapping[str, object] | None) -> Mapping[str, object]:
    """The frozen settings dump inside one record's ``inputs``, or an empty mapping."""
    if not isinstance(inputs, Mapping):
        return {}
    settings = inputs.get("settings")
    resolved = settings.get("resolved") if isinstance(settings, Mapping) else None
    return resolved if isinstance(resolved, Mapping) else {}


def _section(resolved: Mapping[str, object], name: str) -> Mapping[str, object]:
    section = resolved.get(name)
    return section if isinstance(section, Mapping) else {}


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)
