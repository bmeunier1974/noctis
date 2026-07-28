"""Challenger-vs-champion promotion rules — a pure decision function.

Fixed rules, in order:

1. **Activity floor** — reject if fewer than ``min_test_activity`` of the test splits had
   any market exposure. A strategy that almost never trades can post a positive average
   metric on a handful of lucky windows (noise, not edge) and then sit unbeatable at the
   top of the registry. ``0.0`` disables the gate.
2. **Gap guard** — reject if the challenger's train − test metric gap exceeds ``max_gap``
   (an overfit signal), no matter how good the test metric looks.
3. **Forward-holdout gate** — reject if a holdout metric is present and falls below
   ``min_holdout_metric``. The holdout is the most-recent slice the search never touched, so
   this is the backstop against selection/lookahead bias (and web-grounded ideation). Inert
   when the scorecard carries no ``holdout_metric``.
4. **Symbol-holdout gate** — reject if a symbol-holdout metric is present and falls below
   ``min_symbol_holdout_metric``. The held-out symbols were never used in tuning or
   selection, so this is the cross-sectional twin of the temporal gate. Inert when the
   scorecard carries no ``symbol_holdout_metric`` (no symbol-holdout set was reserved).
5. **Consistency gate (optional)** — reject if fewer than ``min_symbol_consistency`` of the
   fit symbols have a positive per-symbol test metric. ``0.0`` disables it (the default —
   a legitimately specialized strategy should not be punished for breadth).
6. **Free slot** — if the registry is below capacity and the challenger clears the minimum
   out-of-sample bar, promote.
7. **Stale champions lose first** — a champion whose scorecard was scored under a different
   metric than the challenger's carries a number in different units; comparing across
   metrics is meaningless. A stale slot behaves like a free slot: any challenger clearing
   the minimum bar displaces the stale champion, and one below the bar is rejected.
8. **Beat the weakest** — otherwise promote iff the challenger's out-of-sample test metric
   beats the weakest champion's; that champion is demoted. On panel scorecards the test
   metric is the panel mean, so champions with different fit sets compare on the same
   scale-free, risk-adjusted footing.

Every outcome carries a one-line rationale for memory and the close report, **and** the
structured evidence behind it: :class:`GateResult` per gate, in the order the gates ran (story
#141). The rationale is for a human; the evidence is for the run record, which has to answer "47
of 66 candidates died at the symbol-holdout gate" — the sentence that makes a results page
credible where an equity curve does not. Nothing about the decision changed to carry it: same
early returns, same order, same outcomes, each gate simply also reporting what it measured.

The evidence is **carried, never consulted**. A number computed for the record may not enter a
decision path (AGENTS.md rule 2), and this module keeps that true by construction: it imports
nothing from the reporting package, and ``tests/test_gate_evidence.py`` proves it in a fresh
subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass

from noctis.backtest.scorecard import Scorecard

# The gates, in the order they run. Every decision's evidence is a **prefix** of this sequence:
# a rejection short-circuits, so it carries the gates reached plus the one that failed, and never
# claims a candidate was measured against a gate it never reached.
GATE_ORDER: tuple[str, ...] = (
    "validated",
    "activity_floor",
    "overfit_gap",
    "reverse_gap",
    "magnitude_cap",
    "forward_holdout",
    "symbol_holdout",
    "symbol_consistency",
)

# The last gate, which of the two depending on the board: a challenger facing a free (or stale)
# slot is judged against the minimum out-of-sample bar; one facing a full board must beat the
# weakest champion. Exactly one of them appears, and only on a decision that cleared everything
# in :data:`GATE_ORDER`.
FINAL_GATES: tuple[str, ...] = ("minimum_bar", "beat_weakest")


@dataclass(frozen=True)
class PromotionRules:
    champion_count: int = 3
    max_gap: float = 1.0
    min_test_metric: float = 0.0
    # Minimum metric on the forward-holdout window; only enforced when the scorecard has one.
    min_holdout_metric: float = 0.0
    # Minimum metric on the held-out symbols; only enforced when the scorecard has one.
    min_symbol_holdout_metric: float = 0.0
    # Minimum fraction of fit symbols with a positive test metric; 0.0 = gate off.
    min_symbol_consistency: float = 0.0
    # Minimum fraction of test splits with market exposure; 0.0 = gate off.
    min_test_activity: float = 0.0
    # Reverse-gap guard: reject when test exceeds train by more than this (a large negative
    # train−test gap is a degeneracy signal, the mirror of the overfit guard); 0.0 = gate off.
    max_reverse_gap: float = 0.0
    # Magnitude sanity cap: reject when |test metric| exceeds this ceiling; 0.0 = gate off.
    max_test_metric: float = 0.0

    @classmethod
    def from_settings(cls, settings) -> PromotionRules:
        """The one mapping from typed config to promotion rules (every entrypoint uses it).

        ``settings`` is duck-typed (anything with ``champion_count`` + ``promotion.*``) so
        this module stays pure — no config import, unit-testable with a stub.
        """
        promotion = settings.promotion
        return cls(
            champion_count=settings.champion_count,
            max_gap=promotion.max_gap,
            min_test_metric=promotion.min_test_metric,
            min_holdout_metric=promotion.min_holdout_metric,
            min_symbol_holdout_metric=promotion.min_symbol_holdout_metric,
            min_symbol_consistency=promotion.min_symbol_consistency,
            min_test_activity=promotion.min_test_activity,
            max_reverse_gap=promotion.max_reverse_gap,
            max_test_metric=promotion.max_test_metric,
        )


@dataclass(frozen=True)
class GateResult:
    """What one gate measured, against what threshold, and whether the candidate got past it.

    The computable half of a verdict. ``observed`` and ``threshold`` are ``None`` where a gate has
    nothing numeric to compare (the validation guard) or the scorecard carried no such metric at
    all (a card with no forward holdout) — a null, never a zero, because "the search never reserved
    a holdout" and "the holdout scored 0" are different facts. ``note`` says why a gate could not
    bite: a threshold of 0 switches it off, and a missing metric makes it inert. Both are still
    *recorded*, because a funnel's denominators are only honest when every gate in the path is on
    the record — "0 of 66 died here" means nothing if the gate might not have run.
    """

    gate: str
    passed: bool
    observed: float | None = None
    threshold: float | None = None
    note: str | None = None


@dataclass(frozen=True)
class Decision:
    promote: bool
    rationale: str
    demote_index: int | None = None  # index into the champions list to demote, if any
    # The evidence behind the verdict, in gate order (story #141). Defaulted, so every existing
    # caller — the registry, the close report, memory — is untouched by its arrival. A rejection
    # short-circuits, so this holds the gates *reached* plus the one that failed; see
    # :data:`GATE_ORDER`.
    gates: tuple[GateResult, ...] = ()


# The half-sentences that say why a gate could not stop anything. Written out here rather than
# inline so the two ways a gate goes inert — switched off by a zero threshold, or handed a metric
# the scorecard never carried — read the same wherever they appear.
_OFF = "inert: switched off by a {name} of 0"
_UNMEASURED = "inert: this scorecard carries no {what} metric, so the gate had nothing to judge"


def decide(challenger: Scorecard, champions: list[Scorecard], rules: PromotionRules) -> Decision:
    """Judge a challenger against the current champions. Pure: scorecards in, decision out.

    The decision is the early-return ladder documented at the top of this module. Beside it, each
    gate appends a :class:`GateResult` to ``decision.gates`` as it evaluates — measurement only,
    read by nothing here.
    """
    gates: list[GateResult] = []
    if challenger.stage != "validated" or not challenger.symbols:
        gates.append(
            GateResult(
                "validated",
                False,
                note=f"stage {challenger.stage!r} with {len(challenger.symbols)} fit symbol(s): "
                "no out-of-sample splits to judge",
            )
        )
        return Decision(
            False,
            "rejected: challenger was not validated (no out-of-sample splits)",
            gates=tuple(gates),
        )
    gates.append(GateResult("validated", True))

    metric = challenger.avg_test_metric
    gap = challenger.gap

    # 1) activity floor — a strategy that almost never trades can post a positive average
    # on a handful of lucky test windows and then sit unbeatable at the top of the registry
    activity = challenger.test_activity
    if rules.min_test_activity > 0.0:
        if activity < rules.min_test_activity:
            gates.append(GateResult("activity_floor", False, activity, rules.min_test_activity))
            return Decision(
                False,
                f"rejected: only {activity:.4f} of test splits traded, below "
                f"{rules.min_test_activity:.2f} (activity floor)",
                gates=tuple(gates),
            )
    gates.append(
        GateResult(
            "activity_floor",
            True,
            activity,
            rules.min_test_activity,
            note=None if rules.min_test_activity > 0.0 else _OFF.format(name="min_test_activity"),
        )
    )

    # 2) overfit guard
    if gap > rules.max_gap:
        gates.append(GateResult("overfit_gap", False, gap, rules.max_gap))
        return Decision(
            False,
            f"rejected: train−test gap {gap:.4f} exceeds max {rules.max_gap:.4f} (overfit)",
            gates=tuple(gates),
        )
    gates.append(GateResult("overfit_gap", True, gap, rules.max_gap))

    # 2b) degeneracy guards — the mirror of overfit and a magnitude sanity cap. A test metric
    # that wildly EXCEEDS train (large negative gap), or an implausibly large test metric, is a
    # noise signal (e.g. a near-zero-downside split annualizing into the thousands) — not edge.
    # Without these a noise scorecard is crowned and, being unbeatable, freezes the registry.
    if rules.max_reverse_gap > 0.0 and -gap > rules.max_reverse_gap:
        gates.append(GateResult("reverse_gap", False, -gap, rules.max_reverse_gap))
        return Decision(
            False,
            f"rejected: test exceeds train by {-gap:.4f}, over max {rules.max_reverse_gap:.4f} "
            f"(degenerate — likely noise, not a robust edge)",
            gates=tuple(gates),
        )
    gates.append(
        GateResult(
            "reverse_gap",
            True,
            -gap,
            rules.max_reverse_gap,
            note=None if rules.max_reverse_gap > 0.0 else _OFF.format(name="max_reverse_gap"),
        )
    )
    if rules.max_test_metric > 0.0 and abs(metric) > rules.max_test_metric:
        gates.append(GateResult("magnitude_cap", False, abs(metric), rules.max_test_metric))
        return Decision(
            False,
            f"rejected: |test metric| {abs(metric):.4f} exceeds sane ceiling "
            f"{rules.max_test_metric:.4f} (degenerate metric)",
            gates=tuple(gates),
        )
    gates.append(
        GateResult(
            "magnitude_cap",
            True,
            abs(metric),
            rules.max_test_metric,
            note=None if rules.max_test_metric > 0.0 else _OFF.format(name="max_test_metric"),
        )
    )

    # 3) forward-holdout gate — must survive on data the search never touched
    holdout = challenger.holdout_metric
    if holdout is not None and holdout < rules.min_holdout_metric:
        gates.append(GateResult("forward_holdout", False, holdout, rules.min_holdout_metric))
        return Decision(
            False,
            f"rejected: holdout metric {holdout:.4f} below bar "
            f"{rules.min_holdout_metric:.4f} (forward-holdout gate)",
            gates=tuple(gates),
        )
    gates.append(
        GateResult(
            "forward_holdout",
            True,
            holdout,
            rules.min_holdout_metric,
            note=None if holdout is not None else _UNMEASURED.format(what="forward-holdout"),
        )
    )

    # 4) symbol-holdout gate — must survive on symbols the search never touched
    symbol_holdout = challenger.symbol_holdout_metric
    if symbol_holdout is not None and symbol_holdout < rules.min_symbol_holdout_metric:
        gates.append(
            GateResult("symbol_holdout", False, symbol_holdout, rules.min_symbol_holdout_metric)
        )
        return Decision(
            False,
            f"rejected: symbol-holdout metric {symbol_holdout:.4f} below bar "
            f"{rules.min_symbol_holdout_metric:.4f} (symbol-holdout gate)",
            gates=tuple(gates),
        )
    gates.append(
        GateResult(
            "symbol_holdout",
            True,
            symbol_holdout,
            rules.min_symbol_holdout_metric,
            note=None if symbol_holdout is not None else _UNMEASURED.format(what="symbol-holdout"),
        )
    )

    # 5) optional breadth gate — fraction of fit symbols with a positive test metric
    per_symbol = challenger.symbol_test_metrics()
    consistency = sum(1 for v in per_symbol.values() if v > 0.0) / len(per_symbol)
    if rules.min_symbol_consistency > 0.0 and challenger.symbols:
        if consistency < rules.min_symbol_consistency:
            gates.append(
                GateResult("symbol_consistency", False, consistency, rules.min_symbol_consistency)
            )
            return Decision(
                False,
                f"rejected: only {consistency:.2f} of fit symbols positive, below "
                f"{rules.min_symbol_consistency:.2f} (symbol-consistency gate)",
                gates=tuple(gates),
            )
    gates.append(
        GateResult(
            "symbol_consistency",
            True,
            consistency,
            rules.min_symbol_consistency,
            note=None
            if rules.min_symbol_consistency > 0.0
            else _OFF.format(name="min_symbol_consistency"),
        )
    )

    # 6) free capacity
    if len(champions) < rules.champion_count:
        free_slot = (
            f"a free slot: the board holds {len(champions)} of {rules.champion_count} champions"
        )
        if metric > rules.min_test_metric:
            gates.append(
                GateResult("minimum_bar", True, metric, rules.min_test_metric, note=free_slot)
            )
            return Decision(
                True,
                f"promoted: test metric {metric:.4f} clears bar {rules.min_test_metric:.4f} "
                f"into a free slot (gap {gap:.4f})",
                gates=tuple(gates),
            )
        gates.append(
            GateResult("minimum_bar", False, metric, rules.min_test_metric, note=free_slot)
        )
        return Decision(
            False,
            f"rejected: test metric {metric:.4f} below minimum bar {rules.min_test_metric:.4f}",
            gates=tuple(gates),
        )

    # 7) stale champions lose first — a champion scored under a different metric carries a
    # number in different units, so its stored value cannot be compared. A stale slot
    # behaves like a free slot: the minimum bar applies, nothing more.
    stale = [i for i, c in enumerate(champions) if c.metric_name != challenger.metric_name]
    if stale:
        stale_slot = (
            f"a stale slot: champion {stale[0]} was scored on "
            f"'{champions[stale[0]].metric_name}', not '{challenger.metric_name}', so only the "
            "minimum bar applies"
        )
        if metric > rules.min_test_metric:
            idx = stale[0]
            gates.append(
                GateResult("minimum_bar", True, metric, rules.min_test_metric, note=stale_slot)
            )
            return Decision(
                True,
                f"promoted: test metric {metric:.4f} clears bar {rules.min_test_metric:.4f}; "
                f"displacing stale champion scored on '{champions[idx].metric_name}' "
                f"(current metric '{challenger.metric_name}')",
                demote_index=idx,
                gates=tuple(gates),
            )
        gates.append(
            GateResult("minimum_bar", False, metric, rules.min_test_metric, note=stale_slot)
        )
        return Decision(
            False,
            f"rejected: test metric {metric:.4f} below minimum bar "
            f"{rules.min_test_metric:.4f} (stale champion present, but the bar still applies)",
            gates=tuple(gates),
        )

    # 8) beat the weakest champion
    weakest_index = min(range(len(champions)), key=lambda i: champions[i].avg_test_metric)
    weakest_metric = champions[weakest_index].avg_test_metric
    weakest_note = (
        f"a full board: the weakest of {len(champions)} champions is "
        f"{champions[weakest_index].family!r}"
    )
    if metric > weakest_metric:
        gates.append(GateResult("beat_weakest", True, metric, weakest_metric, note=weakest_note))
        return Decision(
            True,
            f"promoted: test metric {metric:.4f} beats weakest champion {weakest_metric:.4f}; "
            f"demoting the weakest (gap {gap:.4f})",
            demote_index=weakest_index,
            gates=tuple(gates),
        )
    gates.append(GateResult("beat_weakest", False, metric, weakest_metric, note=weakest_note))
    return Decision(
        False,
        f"rejected: test metric {metric:.4f} does not beat weakest champion {weakest_metric:.4f}",
        gates=tuple(gates),
    )
